import fnmatch
import json
import os

import numpy as np

from utils import whiten


CITYSCAPES_TRAIN_ID_MAP = np.full(256, 255, dtype=np.uint8)
for _label_id, _train_id in (
    (7, 0),
    (8, 1),
    (11, 2),
    (12, 3),
    (13, 4),
    (17, 5),
    (19, 6),
    (20, 7),
    (21, 8),
    (22, 9),
    (23, 10),
    (24, 11),
    (25, 12),
    (26, 13),
    (27, 14),
    (28, 15),
    (31, 16),
    (32, 17),
    (33, 18),
):
    CITYSCAPES_TRAIN_ID_MAP[_label_id] = _train_id


def _one_hot(labels, num_classes):
    return np.eye(num_classes, dtype=np.float32)[labels]


def _load_pil_image(path):
    from PIL import Image

    return Image.open(path)


def _resize_pil_image(image, size, resample):
    if size is None:
        return image
    return image.resize((size[1], size[0]), resample=resample)


def _semantic_to_train_ids(semantic_mask):
    semantic_mask = np.asarray(semantic_mask, dtype=np.uint8)
    return CITYSCAPES_TRAIN_ID_MAP[semantic_mask]


def _decode_cityscapes_disparity(disparity_img):
    disparity = np.asarray(disparity_img, dtype=np.uint16).astype(np.float32)
    valid = disparity > 0
    disparity[valid] = (disparity[valid] - 1.0) / 256.0
    disparity[~valid] = 0.0
    return disparity


def _whiten_sample_major_inputs(x_batch):
    return np.asarray(whiten(np.asarray(x_batch, dtype=np.float64)), dtype=np.float32)


def _flatten_to_mtl(x_batch, target_batches, whiten_inputs=False):
    x_batch = np.asarray(x_batch, dtype=np.float32).reshape(len(x_batch), -1)
    if whiten_inputs:
        x_batch = _whiten_sample_major_inputs(x_batch)

    formatted_targets = []
    for targets in target_batches:
        targets = np.asarray(targets)
        if np.issubdtype(targets.dtype, np.floating):
            targets = targets.astype(np.float32, copy=False)
        formatted_targets.append(targets.reshape(len(targets), -1).T)

    return (x_batch, *formatted_targets)


def get_random_regression_task_MTL(batch_size, in_dim, out_dim_1, out_dim_2):
    X = np.random.randn(batch_size, in_dim)
    Y1 = np.random.randn(batch_size, out_dim_1)
    Y2 = np.random.randn(batch_size, out_dim_2)
    X_whitened = _whiten_sample_major_inputs(X)

    return X_whitened, Y1, Y2


def get_MTL_tasks(
    batch_size,
    in_dim,
    out_dim_1,
    out_dim_2,
    alpha_scales=(1, 1),
    alignment_factor=1,
):
    rng = np.random.default_rng()

    X = np.random.randn(batch_size, in_dim)

    Q_full, _ = np.linalg.qr(rng.standard_normal((in_dim, in_dim)))
    W1 = Q_full[:, :out_dim_1].T
    W2 = (
        alignment_factor * W1
        + (1 - alignment_factor) * Q_full[:, out_dim_1 : 2 * out_dim_1].T
    )
    W2 = W2 / np.linalg.norm(W2, 'fro') * np.linalg.norm(W1, 'fro')
    W1= W1/ np.linalg.norm(W1, 'fro')*np.linalg.norm(W2,'fro')
    X = _whiten_sample_major_inputs(X)
    Y1 = alpha_scales[0] * (W1 @ X.T)
    Y2 = alpha_scales[1] * (W2 @ X.T)

    if out_dim_1 == 1:
        Y1 = Y1.reshape(1, batch_size)
        Y2 = Y2.reshape(1, batch_size)

    return X, Y1, Y2


class MultiMNISTGenerator:
    def __init__(
        self,
        data,
        labels,
        train=True,
        shift_range=4,
        pairs_per_image=10,
        alignment=True,
        max_iou=1,
        corrupt_p=0.0,
        random=None,
    ):
        n_samples = min(50000 if train else 1000, len(data))
        if train:
            self.data = data[:n_samples]
            self.labels = labels[:n_samples]
        else:
            self.data = data[-n_samples:]
            self.labels = labels[-n_samples:]

        self.shift_range = shift_range
        self.pairs_per_image = pairs_per_image
        self.base_size = 28
        self.multi_size = 36
        self.alignment = alignment
        self.max_iou = max_iou
        self.corrupt_p = corrupt_p
        self.random = np.random.default_rng(random)

        self.by_class = {c: np.where(self.labels == c)[0] for c in range(10)}
        self.total = len(self.data) * pairs_per_image

        if not self.alignment:
            q, _ = np.linalg.qr(
                self.random.standard_normal((self.multi_size, self.multi_size))
            )
            self.R = q
        else:
            self.R = np.eye(self.multi_size)

    def __len__(self):
        return self.total

    def __iter__(self):
        for idx in range(self.total):
            img_idx = idx // self.pairs_per_image
            img1 = self.data[img_idx].reshape(self.base_size, self.base_size)
            y1 = int(self.labels[img_idx])

            other_classes = [c for c in range(10) if c != y1]
            y2 = int(self.random.choice(other_classes))
            partner_idx = int(self.random.choice(self.by_class[y2]))
            img2 = self.data[partner_idx].reshape(self.base_size, self.base_size)

            if self.random.random() < self.corrupt_p:
                wrong_choices = [c for c in range(10) if c != y2]
                y2 = int(self.random.choice(wrong_choices))

            canvas1 = np.zeros((self.multi_size, self.multi_size), dtype=np.float32)
            canvas2 = np.zeros((self.multi_size, self.multi_size), dtype=np.float32)

            sh1, sh2 = self._sample_shifts(self.max_iou, self.shift_range)
            canvas1[
                sh1[0] : sh1[0] + self.base_size, sh1[1] : sh1[1] + self.base_size
            ] = img1 / 255.0
            canvas2[
                sh2[0] : sh2[0] + self.base_size, sh2[1] : sh2[1] + self.base_size
            ] = img2 / 255.0

            if not self.alignment:
                canvas2 = self.R @ canvas2 @ self.R.T

            combined = np.maximum(canvas1, canvas2)
            yield combined, (y1, y2)

    def _iou(self, sh1, sh2):
        dy = max(0, self.base_size - abs(sh1[0] - sh2[0]))
        dx = max(0, self.base_size - abs(sh1[1] - sh2[1]))
        inter = dy * dx
        union = 2 * self.base_size * self.base_size - inter
        return inter / union

    def _sample_shifts(self, max_iou, shift_range):
        while True:
            sh1 = self.random.integers(0, shift_range + 1, 2)
            sh2 = self.random.integers(0, shift_range + 1, 2)
            if self._iou(sh1, sh2) <= max_iou:
                return sh1, sh2

    def batchmaker(self, batch_size, generator=None, whiten_inputs=True):
        if generator is None:
            generator = iter(self)

        x_train = []
        y1_train = []
        y2_train = []
      
        for i, (img, (y1, y2)) in enumerate(generator):
            x_train.append(img.reshape(self.multi_size * self.multi_size))
            y1_train.append(_one_hot(y1, 10))
            y2_train.append(_one_hot(y2, 10))
            if i + 1 >= batch_size:
                break
     
        x_train = np.asarray(x_train, dtype=np.float32)
        print("before whitening")
        print("global mean:", x_train.mean())
        print("mean abs feature mean:", np.abs(x_train.mean(axis=0)).mean())
        print("max abs feature mean:", np.abs(x_train.mean(axis=0)).max())

        if whiten_inputs:
          x_train = _whiten_sample_major_inputs(x_train)

        print("after whitening")
        print("global mean:", x_train.mean())
        print("mean abs feature mean:", np.abs(x_train.mean(axis=0)).mean())
        print("max abs feature mean:", np.abs(x_train.mean(axis=0)).max())

        y1_train = np.asarray(y1_train, dtype=np.float32).T
        y2_train = np.asarray(y2_train, dtype=np.float32).T
      

        return x_train, y1_train, y2_train


def get_MultiMNIST_tasks(
    data,
    labels,
    batch_size,
    train=True,
    shift_range=4,
    pairs_per_image=10,
    alignment=True,
    max_iou=1,
    corrupt_p=0.0,
    random=None,
    whiten_inputs=True,
):
    generator = MultiMNISTGenerator(
        data=data,
        labels=labels,
        train=train,
        shift_range=shift_range,
        pairs_per_image=pairs_per_image,
        alignment=alignment,
        max_iou=max_iou,
        corrupt_p=corrupt_p,
        random=random,
    )
    return generator.batchmaker(batch_size, whiten_inputs=whiten_inputs)


get_multimnist_tasks = get_MultiMNIST_tasks


class CityscapesMTLGenerator:
    def __init__(
        self,
        root,
        split="train",
        mode="fine",
        resize=None,
        semantic_train_ids=True,
        disparity_normalize=True,
    ):
        from torchvision.datasets import Cityscapes

        self.dataset = Cityscapes(
            root=root,
            split=split,
            mode=mode,
            target_type="semantic",
        )
        self.resize = resize
        self.semantic_train_ids = semantic_train_ids
        self.disparity_normalize = disparity_normalize

    def __len__(self):
        return len(self.dataset)

    def _get_disparity_path(self, index):
        image_path = self.dataset.images[index]
        return (
            image_path.replace("leftImg8bit", "disparity")
            .replace("_leftImg8bit.png", "_disparity.png")
        )

    def __getitem__(self, index):
        from PIL import Image

        image, semantic = self.dataset[index]
        disparity = _load_pil_image(self._get_disparity_path(index))

        image = _resize_pil_image(image, self.resize, Image.BILINEAR)
        semantic = _resize_pil_image(semantic, self.resize, Image.NEAREST)
        disparity = _resize_pil_image(disparity, self.resize, Image.NEAREST)

        image = np.asarray(image, dtype=np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        if self.semantic_train_ids:
            semantic = _semantic_to_train_ids(semantic)
        else:
            semantic = np.asarray(semantic, dtype=np.int64)

        disparity = _decode_cityscapes_disparity(disparity)
        if self.disparity_normalize:
            max_disparity = np.max(disparity)
            if max_disparity > 0:
                disparity = disparity / max_disparity

        disparity = disparity[np.newaxis, ...]

        return image, semantic.astype(np.int64), disparity.astype(np.float32)

    def batchmaker(self, batch_size, start_idx=0):
        x_batch = []
        semantic_batch = []
        disparity_batch = []

        end_idx = min(start_idx + batch_size, len(self))
        for idx in range(start_idx, end_idx):
            image, semantic, disparity = self[idx]
            x_batch.append(image)
            semantic_batch.append(semantic)
            disparity_batch.append(disparity)

        x_batch = np.asarray(x_batch, dtype=np.float32)
        semantic_batch = np.asarray(semantic_batch, dtype=np.int64)
        disparity_batch = np.asarray(disparity_batch, dtype=np.float32)

        return x_batch, semantic_batch, disparity_batch


def get_Cityscapes_tasks(
    root,
    batch_size,
    split="train",
    mode="fine",
    resize=None,
    semantic_train_ids=True,
    disparity_normalize=True,
    start_idx=0,
):
    generator = CityscapesMTLGenerator(
        root=root,
        split=split,
        mode=mode,
        resize=resize,
        semantic_train_ids=semantic_train_ids,
        disparity_normalize=disparity_normalize,
    )
    return generator.batchmaker(batch_size=batch_size, start_idx=start_idx)


get_cityscapes_tasks = get_Cityscapes_tasks


class RandomScaleCrop:
    def __init__(self, scale=(1.0, 1.2, 1.5), random=None):
        self.scale = scale
        self.random = random if random is not None else np.random.default_rng()

    def __call__(self, img, label, depth, normal):
        import torch
        import torch.nn.functional as F

        height, width = img.shape[-2:]
        sc = self.scale[int(self.random.integers(0, len(self.scale)))]
        h, w = int(height / sc), int(width / sc)
        i = int(self.random.integers(0, height - h + 1))
        j = int(self.random.integers(0, width - w + 1))

        img = F.interpolate(
            img[None, :, i : i + h, j : j + w],
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
        label = F.interpolate(
            label[None, None, i : i + h, j : j + w],
            size=(height, width),
            mode="nearest",
        ).squeeze(0).squeeze(0)
        depth = F.interpolate(
            depth[None, :, i : i + h, j : j + w],
            size=(height, width),
            mode="nearest",
        ).squeeze(0)
        normal = F.interpolate(
            normal[None, :, i : i + h, j : j + w],
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)

        return img, label, depth / sc, normal


class NYUv2MTLGenerator:
    def __init__(
        self,
        root,
        mode="train",
        augmentation=False,
        split_json=None,
        tasks=("segmentation", "depth"),
        random=None,
    ):
        self.mode = mode
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation
        self.tasks = tasks
        self.random = np.random.default_rng(random)

        if split_json is None:
            split_json = os.path.join(self.root, "data_split.json")

        if mode in {"train", "val", "trainval"}:
            if os.path.exists(split_json):
                with open(split_json, "r", encoding="utf-8") as f:
                    data_split = json.load(f)
                train_index = data_split["train"]
                val_index = data_split["val"]
            else:
                val_index=[3, 4, 10, 523, 524, 532, 26, 27, 539, 36, 45, 46, 561, 55, 56, 569, 58, 66, 581, 583, 74, 75, 586, 588, 81, 82, 596, 599, 601, 91, 92, 93, 94, 99, 106, 618, 109, 623, 113, 114, 625, 631, 120, 632, 122, 636, 127, 639, 131, 644, 645, 653, 144, 656, 148, 661, 157, 669, 672, 161, 673, 165, 168, 170, 171, 682, 684, 692, 182, 696, 185, 186, 187, 697, 702, 704, 708, 202, 714, 204, 208, 209, 720, 212, 724, 730, 732, 221, 737, 227, 228, 740, 230, 240, 241, 242, 243, 249, 254, 766, 256, 767, 264, 780, 271, 783, 273, 791, 284, 285, 294, 296, 297, 298, 299, 302, 304, 307, 320, 326, 329, 337, 339, 353, 362, 365, 370, 371, 373, 375, 376, 378, 381, 387, 391, 397, 399, 403, 404, 406, 407, 417, 423, 424, 433, 448, 453, 465, 472, 474, 484, 486, 493, 494, 495, 496, 499, 501, 506]
                train_index=[43, 739, 300, 278, 180, 19, 330, 364, 519, 158, 52, 762, 38, 633, 184, 476, 584, 123, 235, 30, 530, 181, 721, 460, 267, 49, 215, 627, 735, 640, 624, 100, 441, 480, 172, 32, 517, 369, 80, 635, 556, 361, 253, 133, 778, 548, 367, 564, 543, 388, 772, 87, 434, 600, 634, 510, 306, 668, 768, 572, 11, 622, 425, 619, 535, 175, 679, 281, 73, 638, 319, 90, 577, 566, 483, 683, 562, 745, 518, 630, 520, 150, 628, 126, 503, 142, 343, 24, 477, 654, 69, 321, 237, 440, 789, 563, 366, 710, 648, 482, 132, 316, 372, 420, 59, 641, 455, 151, 660, 587, 651, 205, 313, 78, 86, 166, 252, 57, 478, 610, 751, 701, 356, 224, 409, 666, 681, 268, 146, 567, 20, 137, 706, 707, 546, 85, 121, 418, 606, 33, 398, 437, 396, 15, 160, 352, 314, 198, 607, 135, 514, 726, 318, 111, 155, 473, 200, 675, 664, 89, 438, 491, 213, 179, 77, 749, 98, 247, 325, 674, 585, 468, 211, 513, 188, 190, 790, 244, 700, 105, 578, 250, 658, 536, 786, 770, 413, 25, 344, 688, 348, 309, 756, 531, 487, 620, 234, 516, 116, 328, 236, 368, 338, 415, 83, 769, 727, 731, 788, 110, 112, 611, 576, 395, 347, 509, 412, 459, 303, 29, 402, 593, 1, 647, 196, 84, 449, 782, 549, 164, 140, 226, 728, 217, 216, 259, 580, 22, 529, 269, 589, 777, 39, 411, 511, 400, 575, 401, 550, 680, 263, 629, 526, 9, 779, 612, 289, 232, 139, 613, 671, 104, 489, 44, 590, 60, 504, 42, 207, 162, 2, 475, 129, 147, 559, 545, 505, 342, 384, 422, 282, 657, 522, 698, 290, 334, 414, 464, 35, 416, 785, 512, 764, 408, 752, 678, 21, 582, 703, 722, 143, 551, 662, 677, 637, 257, 574, 775, 533, 426, 195, 686, 210, 659, 450, 663, 500, 492, 705, 193, 47, 358, 323, 124, 63, 442, 17, 763, 119, 753, 118, 760, 784, 676, 670, 152, 222, 291, 174, 431, 332, 197, 689, 18, 13, 333, 61, 239, 283, 498, 744, 711, 359, 665, 615, 521, 363, 410, 621, 755, 713, 199, 394, 461, 570, 781, 758, 773, 349, 542, 383, 163, 231, 138, 311, 427, 444, 471, 277, 719, 292, 136, 568, 266, 262, 238, 206, 280, 757, 446, 733, 507, 429, 738, 723, 346, 102, 276, 103, 774, 597, 97, 392, 603, 65, 107, 156, 690, 485, 419, 229, 261, 248, 177, 335, 605, 141, 687, 37, 79, 67, 754, 443, 558, 565, 219, 591, 377, 579, 571, 655, 154, 602, 421, 246, 469, 718, 458, 592, 14, 712, 245, 385, 23, 7, 134, 350, 553, 255, 761, 541, 130, 598, 295, 456, 479, 183, 716, 488, 70, 322, 173, 557, 390, 5, 341, 31, 642, 435, 51, 609, 750, 646, 260, 497, 53, 324, 380, 108, 691, 405, 463, 776, 355, 534, 301, 452, 194, 573, 389, 537, 595, 286, 447, 515, 0, 502, 169, 462, 34, 552, 508, 439, 101, 153, 725, 466, 95, 287, 315, 430, 203, 594, 178, 72, 555, 792, 695, 115, 50, 604, 538, 428, 41, 743, 650, 528, 787, 379, 68, 275, 643, 525, 96, 28, 265, 748, 685, 560, 617, 746, 481, 345, 432, 251, 490, 48, 717, 393, 167, 223, 8, 201, 544, 189, 305, 62, 233, 40, 793, 340, 220, 54, 382, 351, 270, 457, 176, 317, 327, 374, 626, 218, 258, 274, 694, 225, 616, 279, 214, 715, 736, 451, 527, 76, 445, 649, 554, 771, 16, 149, 454, 12, 357, 540, 547, 699, 467, 360, 386, 6, 312, 765, 192, 734, 794, 747, 709, 191, 336, 759, 288, 125, 159, 741, 436, 310, 608, 88, 354, 71, 272, 729, 614, 308, 293, 742, 652, 117, 470, 145, 128, 693, 667, 331, 64]

            if mode == "train":
                self.index_list = train_index
            elif mode == "val":
                self.index_list = val_index
            else:
                self.index_list = train_index + val_index
            self.data_path = os.path.join(self.root, "train")
        elif mode == "test":
            image_dir = os.path.join(self.root, "val", "image")
            data_len = len(fnmatch.filter(os.listdir(image_dir), "*.npy"))
            self.index_list = list(range(data_len))
            self.data_path = os.path.join(self.root, "val")
        else:
            raise ValueError(f"Unsupported NYUv2 mode: {mode}")

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, i):
        import torch

        index = self.index_list[i]
        image = torch.from_numpy(
            np.moveaxis(
                np.load(os.path.join(self.data_path, "image", f"{index}.npy")),
                -1,
                0,
            )
        )
        semantic = torch.from_numpy(
            np.load(os.path.join(self.data_path, "label", f"{index}.npy"))
        )
        depth = torch.from_numpy(
            np.moveaxis(
                np.load(os.path.join(self.data_path, "depth", f"{index}.npy")),
                -1,
                0,
            )
        )
        normal = torch.from_numpy(
            np.moveaxis(
                np.load(os.path.join(self.data_path, "normal", f"{index}.npy")),
                -1,
                0,
            )
        )

        if self.augmentation:
            augmenter = RandomScaleCrop(random=self.random)
            image, semantic, depth, normal = augmenter(image, semantic, depth, normal)
            if self.random.random() < 0.5:
                image = torch.flip(image, dims=[2])
                semantic = torch.flip(semantic, dims=[1])
                depth = torch.flip(depth, dims=[2])
                normal = torch.flip(normal, dims=[2])
                normal[0, :, :] = -normal[0, :, :]

        targets = {
            "segmentation": semantic.float(),
            "depth": depth.float(),
            "normal": normal.float(),
        }

        return image.float(), targets

    def batchmaker(self, batch_size, start_idx=0, mtl_format=False, whiten_inputs=False):
        x_batch = []
        target_batches = {task: [] for task in self.tasks}

        end_idx = min(start_idx + batch_size, len(self))
        for idx in range(start_idx, end_idx):
            image, targets = self[idx]
            x_batch.append(image.numpy())
            for task in self.tasks:
                target_batches[task].append(targets[task].numpy())

        ordered_targets = [target_batches[task] for task in self.tasks]

        if mtl_format:
            return _flatten_to_mtl(
                x_batch,
                ordered_targets,
                whiten_inputs=whiten_inputs,
            )

        x_batch = np.asarray(x_batch, dtype=np.float32)
        outputs = [x_batch]

        for task in self.tasks:
            if task == "segmentation":
                outputs.append(np.asarray(target_batches[task], dtype=np.int64))
            else:
                outputs.append(np.asarray(target_batches[task], dtype=np.float32))

        return tuple(outputs)


def get_NYUv2_tasks(
    root,
    batch_size,
    mode="train",
    augmentation=False,
    split_json=None,
    tasks=("segmentation", "depth"),
    start_idx=0,
    random=None,
    mtl_format=False,
    whiten_inputs=False,
):
    generator = NYUv2MTLGenerator(
        root=root,
        mode=mode,
        augmentation=augmentation,
        split_json=split_json,
        tasks=tasks,
        random=random,
    )
    return generator.batchmaker(
        batch_size=batch_size,
        start_idx=start_idx,
        mtl_format=mtl_format,
        whiten_inputs=whiten_inputs,
    )


get_nyuv2_tasks = get_NYUv2_tasks
get_MTL_Tasks = get_MTL_tasks
