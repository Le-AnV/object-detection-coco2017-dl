import torch
import torch.nn as nn
import math


class ComputeLoss(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        # Stride tương ứng các feature map P3, P4, P5
        self.strides = [8, 16, 32]

        # Hệ số trọng số cho từng thành phần loss
        self.box_gain = 0.1
        self.cls_gain = 0.5
        self.obj_gain = 0.7
        self.balance = [3.0, 1.0, 0.4]

        self.BCEcls = nn.BCEWithLogitsLoss(reduction="none")
        self.BCEobj = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds, targets):
        """
        preds: danh sách [P3, P4, P5]
               mỗi phần tử shape [B, 5 + num_classes, H, W]
        targets: Tensor [N, 6] = batch_id, class, cx, cy, w, h (đã chuẩn hóa)
        """
        device = preds[0].device
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        lcls = torch.zeros(1, device=device)

        tcls, tbox, indices = self.build_targets(preds, targets)

        for i, pi in enumerate(preds):
            # Đưa tensor về dạng [B, 1, H, W, C]
            pi = pi.permute(0, 2, 3, 1).unsqueeze(1)

            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 4], device=device)

            n = b.shape[0]
            if n > 0:
                ps = pi[b, a, gj, gi]

                # Loss hồi quy bounding box
                pxy = ps[:, 0:2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2.0) ** 2
                pbox = torch.cat((pxy, pwh), dim=1)

                iou = self.bbox_iou(pbox, tbox[i], CIoU=True)
                lbox += (1.0 - iou).mean()

                # Gán nhãn objectness theo IoU
                tobj[b, a, gj, gi] = iou.detach().clamp(0).to(tobj.dtype)

                # Loss phân lớp
                if self.num_classes > 1:
                    t = torch.zeros_like(ps[:, 5:], device=device)
                    t[torch.arange(n), tcls[i]] = 1.0
                    lcls += self.BCEcls(ps[:, 5:], t).mean()

            # Loss objectness trên toàn feature map
            obj_loss = self.BCEobj(pi[..., 4], tobj)
            lobj += obj_loss.mean() * self.balance[i]

        # Tổng hợp loss
        total_loss = lbox * self.box_gain + lobj * self.obj_gain + lcls * self.cls_gain

        return total_loss, {
            "box": lbox.detach(),
            "obj": lobj.detach(),
            "cls": lcls.detach(),
        }

    # ============================================================
    # GÁN NHÃN TARGET (ANCHOR-FREE + OFFSET)
    # ============================================================
    def build_targets(self, preds, targets):
        tcls, tbox, indices = [], [], []

        device = targets.device

        # Các offset cho ô lân cận
        offsets = torch.tensor(
            [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
            device=device,
            dtype=torch.long,
        )

        for i, stride in enumerate(self.strides):
            _, _, h, w = preds[i].shape
            gain = torch.tensor([1, 1, w, h, w, h], device=device)

            if targets.shape[0] == 0:
                tcls.append(torch.tensor([], device=device, dtype=torch.long))
                tbox.append(torch.tensor([], device=device))
                indices.append(
                    (
                        torch.tensor([], device=device, dtype=torch.long),
                        torch.tensor([], device=device, dtype=torch.long),
                        torch.tensor([], device=device, dtype=torch.long),
                        torch.tensor([], device=device, dtype=torch.long),
                    )
                )
                continue

            t = targets * gain
            b = t[:, 0].long()
            c = t[:, 1].long()
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]

            gij = gxy.long()

            bi, gi, gj, tbox_i, tcls_i = [], [], [], [], []

            for off in offsets:
                gij_o = gij + off
                gi_o, gj_o = gij_o[:, 0], gij_o[:, 1]

                mask = (gi_o >= 0) & (gj_o >= 0) & (gi_o < w) & (gj_o < h)
                if mask.sum() == 0:
                    continue

                bi.append(b[mask])
                gi.append(gi_o[mask])
                gj.append(gj_o[mask])
                tbox_i.append(
                    torch.cat(
                        (gxy[mask] - gij_o[mask], gwh[mask]),
                        dim=1,
                    )
                )
                tcls_i.append(c[mask])

            if len(bi):
                bi = torch.cat(bi)
                gi = torch.cat(gi)
                gj = torch.cat(gj)
                tbox_i = torch.cat(tbox_i)
                tcls_i = torch.cat(tcls_i)
                a = torch.zeros_like(bi)

                indices.append((bi, a, gj, gi))
                tbox.append(tbox_i)
                tcls.append(tcls_i)
            else:
                tcls.append(torch.tensor([], device=device, dtype=torch.long))
                tbox.append(torch.tensor([], device=device))
                indices.append(
                    (
                        torch.tensor([], device=device, dtype=torch.long),
                        torch.tensor([], device=device, dtype=torch.long),
                        torch.tensor([], device=device, dtype=torch.long),
                        torch.tensor([], device=device, dtype=torch.long),
                    )
                )

        return tcls, tbox, indices

    # ============================================================
    # TÍNH IOU (CIOU)
    # ============================================================
    @staticmethod
    def bbox_iou(box1, box2, CIoU=True, eps=1e-7):
        # Chuyển từ cxcywh sang xyxy
        b1_x1 = box1[:, 0] - box1[:, 2] / 2
        b1_y1 = box1[:, 1] - box1[:, 3] / 2
        b1_x2 = box1[:, 0] + box1[:, 2] / 2
        b1_y2 = box1[:, 1] + box1[:, 3] / 2

        b2_x1 = box2[:, 0] - box2[:, 2] / 2
        b2_y1 = box2[:, 1] - box2[:, 3] / 2
        b2_x2 = box2[:, 0] + box2[:, 2] / 2
        b2_y2 = box2[:, 1] + box2[:, 3] / 2

        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
            torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
        ).clamp(0)

        w1, h1 = box1[:, 2], box1[:, 3]
        w2, h2 = box2[:, 2], box2[:, 3]

        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union

        if CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
            c2 = cw**2 + ch**2 + eps

            rho2 = (box2[:, 0] - box1[:, 0]) ** 2 + (box2[:, 1] - box1[:, 1]) ** 2

            v = 4 / math.pi**2 * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

            with torch.no_grad():
                alpha = v / (v - iou + 1 + eps)

            return iou - (rho2 / c2 + alpha * v)

        return iou
