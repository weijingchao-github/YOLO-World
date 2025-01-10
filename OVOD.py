import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)

import copy
from types import SimpleNamespace

import cv2
import numpy as np
import rospy
import supervision as sv
import torch
from cv_bridge import CvBridge
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmengine.config import Config
from mmengine.dataset import Compose
from sensor_msgs.msg import Image
from torchvision.ops import nms


class YoloWorld:

    class_names = (
        "person, cup, chair, desk, screen monitor, laptop, mouse, keyboard, "
        "cell phone, book, wrist watch, pen, bottle, bag, squre lamp, computer case, human face "
        # "person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, "
        # "traffic light, fire hydrant, stop sign, parking meter, bench, bird, "
        # "cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, "
        # "backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, "
        # "sports ball, kite, baseball bat, baseball glove, skateboard, "
        # "surfboard, tennis racket, bottle, wine glass, cup, fork, knife, "
        # "spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, "
        # "hot dog, pizza, donut, cake, chair, couch, potted plant, bed, "
        # "dining table, toilet, tv, laptop, mouse, remote, keyboard, "
        # "cell phone, microwave, oven, toaster, sink, refrigerator, book, "
        # "clock, vase, scissors, teddy bear, hair drier, toothbrush, "
        # "watch "
        # "desk, chair, filing cabinet, bookshelf, conference table, whiteboard, bulletin board, drawer, couch, pen, pencil, eraser, ruler, paperclip, stapler, staples, tape, glue, scissors, highlighter, marker, notebook, folder, binder, envelopes, post-it notes, sticky notes, computer, laptop, monitor, keyboard, mouse, printer, scanner, photocopier, projector, speaker, microphone, headphones, earphones, telephone, calculator, power strip, document, report, invoice, contract, proposal, memo, brochure, manual, calendar, schedule, coffee machine, water dispenser, trash bin, wastebasket, shredder, clock, air conditioner, fan, curtain "
    )

    def __init__(self):
        # model init
        config_path = os.path.join(
            os.path.dirname(__file__),
            "configs/pretrain/yolo_world_v2_l_clip_large_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_800ft_lvis_minival.py",
        )
        checkpoint_path = os.path.join(
            os.path.dirname(__file__),
            "pretrained_weights/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-9df82e55.pth",
        )
        cfg = Config.fromfile(config_path)
        cfg.work_dir = os.path.join(os.path.dirname(__file__))
        cfg.load_from = checkpoint_path
        cfg.test_dataloader.dataset.dataset.data_root = os.path.join(
            os.path.dirname(__file__),
            "data/coco",
        )
        cfg.test_dataloader.dataset.class_text_path = os.path.join(
            os.path.dirname(__file__),
            "data/texts/lvis_v1_class_texts.json",
        )
        self.model = init_detector(cfg, checkpoint=checkpoint_path, device="cuda:0")
        # others
        self.algs_args = SimpleNamespace(score_thr=0.2, max_num_boxes=100, nms_thr=0.5)
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        test_pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
        self.test_pipeline = Compose(test_pipeline_cfg)
        self.classes = []
        classes_str = YoloWorld.class_names.split(",")
        for class_name in classes_str:
            class_name = class_name.strip()
            self.classes.append([class_name])
        self.person_tracker = sv.ByteTrack()
        self.enable_vis = True
        if self.enable_vis:
            self.bounding_box_annotator = sv.BoxAnnotator()
            self.label_annotator = sv.LabelAnnotator()
        # ROS init
        self.bridge = CvBridge()
        rospy.Subscriber(
            "/camera/color/image_raw",
            Image,
            self._ov_object_detect,
            queue_size=1,
        )

    def _ov_object_detect(self, image_color):
        image_color = self.bridge.imgmsg_to_cv2(image_color, desired_encoding="bgr8")
        # cv2.imshow("i", image_color)
        # cv2.waitKey(1)
        image_show = copy.deepcopy(image_color)
        image_color = image_color[:, :, [2, 1, 0]]
        data_info = dict(img=image_color, img_id=0, texts=self.classes)
        data_info = self.test_pipeline(data_info)
        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0),
            data_samples=[data_info["data_samples"]],
        )
        with torch.no_grad():
            output = self.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        # nms
        keep_idxs = nms(
            pred_instances.bboxes,
            pred_instances.scores,
            iou_threshold=self.algs_args.nms_thr,
        )
        pred_instances = pred_instances[keep_idxs]
        # score thresholding
        pred_instances = pred_instances[
            pred_instances.scores.float() > self.algs_args.score_thr
        ]
        # max detections
        if len(pred_instances.scores) > self.algs_args.max_num_boxes:
            indices = pred_instances.scores.float().topk(self.algs_args.max_num_boxes)[
                1
            ]
            pred_instances = pred_instances[indices]
        pred_instances = pred_instances.cpu().numpy()
        # 就算没有检测到bbox，下面这行代码也不会报错，只不过里面的数据都是空的，可以遮住摄像头debug看看
        detections_all = sv.Detections(
            xyxy=pred_instances["bboxes"],
            class_id=pred_instances["labels"],
            confidence=pred_instances["scores"],
        )
        detections_person = {"bboxes": [], "labels": [], "scores": []}
        detections_others = {"bboxes": [], "labels": [], "scores": []}
        for index, class_id in enumerate(pred_instances["labels"]):
            if class_id == 0:
                detections_person["bboxes"].append(pred_instances["bboxes"][index])
                detections_person["labels"].append(pred_instances["labels"][index])
                detections_person["scores"].append(pred_instances["scores"][index])
            else:
                detections_others["bboxes"].append(pred_instances["bboxes"][index])
                detections_others["labels"].append(pred_instances["labels"][index])
                detections_others["scores"].append(pred_instances["scores"][index])
        if len(detections_person["bboxes"]) == 0:
            detections_person = None
        else:
            detections_person = sv.Detections(
                xyxy=np.array(detections_person["bboxes"]),
                class_id=np.array(detections_person["labels"]),
                confidence=np.array(detections_person["scores"]),
            )
        if len(detections_others["bboxes"]) == 0:
            detections_others = None
        else:
            detections_others = sv.Detections(
                xyxy=np.array(detections_others["bboxes"]),
                class_id=np.array(detections_others["labels"]),
                confidence=np.array(detections_others["scores"]),
            )
        # update tracker
        if detections_person is not None:
            detections_person = self.person_tracker.update_with_detections(
                detections_person
            )
        else:
            self.person_tracker.update_with_tensors(np.zeros((1, 5)))
        if self.enable_vis:
            self._visualize(
                image_show, detections_all, detections_person, detections_others
            )

    def _visualize(
        self, image_show, detections_all, detections_person, detections_others
    ):
        # draw bboxes
        image_show = self.bounding_box_annotator.annotate(image_show, detections_all)
        # person: tracker number
        if detections_person is not None:
            person_labels = [
                f"{tracker_id}" for tracker_id in detections_person.tracker_id
            ]
            image_show = self.label_annotator.annotate(
                image_show, detections_person, person_labels
            )
        # others: label name with confidence score
        if detections_others is not None:
            others_labels = [
                f"{self.classes[class_id][0]} {confidence:0.2f}"
                for class_id, confidence in zip(
                    detections_others.class_id, detections_others.confidence
                )
            ]
            image_show = self.label_annotator.annotate(
                image_show, detections_others, others_labels
            )

        cv2.imshow("OVOD", image_show)
        cv2.waitKey(1)


def main():
    rospy.init_node("open_vocabulary_object_detect")
    YoloWorld()
    rospy.spin()


if __name__ == "__main__":
    main()
