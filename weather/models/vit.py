from torch import nn
from transformers import ViTForImageClassification, ViTImageProcessor


class ViTWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, output_attentions=False):
        outputs = self.model(x, output_attentions=output_attentions)
        if output_attentions:
            return outputs
        else:
            return outputs.logits


def get_vit_model(class_names, device, model_name="google/vit-base-patch16-224", output_attentions=False):
    processor = ViTImageProcessor.from_pretrained(model_name)

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(class_names),
        id2label={i: c for i, c in enumerate(class_names)},
        label2id={c: i for i, c in enumerate(class_names)},
        ignore_mismatched_sizes=True,
        output_attentions=output_attentions,
        output_hidden_states=True
    )

    wrapped_model = ViTWrapper(model).to(device)

    vit_info = {
        'mean': processor.image_mean,
        'std': processor.image_std,
        'size': processor.size['height']
    }

    return wrapped_model, vit_info
