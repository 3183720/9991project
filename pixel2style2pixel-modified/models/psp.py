"""
This file defines the core research contribution
"""
import matplotlib

matplotlib.use("Agg")
import torch
from torch import nn
from models.encoders import psp_encoders
from models.stylegan2.model import Generator
from configs.paths_config import model_paths
from utils.model_utils import RESNET_MAPPING

def get_keys(d, name):
    if "state_dict" in d:
        d = d["state_dict"]
    d_filt = {
        k[len(name) + 1 :]: v for k, v in d.items() if k[: len(name)] == name
    }
    return d_filt


class pSp(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.set_opts(opts)
        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = Generator(
            self.opts.output_size, 512, 8, channel_multiplier=2, c_dim=self.opts.c_dim
        )

        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def set_encoder(self):
        if self.opts.encoder_type == "GradualStyleEncoder":
            encoder = psp_encoders.GradualStyleEncoder(50, "ir_se", self.opts)
        elif self.opts.encoder_type == "BackboneEncoderUsingLastLayerIntoW":
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(
                50, "ir_se", self.opts
            )
        elif (
            self.opts.encoder_type == "BackboneEncoderUsingLastLayerIntoWPlus"
        ):
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(
                50, "ir_se", self.opts
            )
        elif self.opts.encoder_type == 'ResNetGradualStyleEncoder':
            encoder = psp_encoders.ResNetGradualStyleEncoder(opts=self.opts )
        else:
            raise Exception(
                "{} is not a valid encoders".format(self.opts.encoder_type)
            )
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print(
                "Loading pSp from checkpoint: {}".format(
                    self.opts.checkpoint_path
                )
            )
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
            self.encoder.load_state_dict(
                get_keys(ckpt, "encoder"), strict=True
            )
            self.decoder.load_state_dict(
                get_keys(ckpt, "decoder"), strict=True
            )
            self.__load_latent_avg(ckpt)
        else:
            print("Loading encoders weights from resnet34!")
            encoder_ckpt = torch.load(model_paths['resnet34'])
            
            # if input to encoder is not an RGB image, do not load the input layer weights
            # if self.opts.label_nc != 0:
            #     encoder_ckpt = {
            #         k: v
            #         for k, v in encoder_ckpt.items()
            #         if "input_layer" not in k
            #     }
            encoder_ckpt = self.__get_encoder_checkpoint()
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print("Loading decoder weights from pretrained!")
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
            if self.opts.learn_in_w:
                self.__load_latent_avg(ckpt, repeat=1)
            else:
                self.__load_latent_avg(ckpt, repeat=18)


    def __get_encoder_checkpoint(self):
      # if "ffhq" in self.opts.dataset_type:
      # 	print('Loading encoders weights from irse50!')
      # 	encoder_ckpt = torch.load(model_paths['ir_se50'])
      # 	# Transfer the RGB input of the irse50 network to the first 3 input channels of pSp's encoder
      # 	if self.opts.input_nc != 3:
      # 		shape = encoder_ckpt['input_layer.0.weight'].shape
      # 		altered_input_layer = torch.randn(shape[0], self.opts.input_nc, shape[2], shape[3], dtype=torch.float32)
      # 		altered_input_layer[:, :3, :, :] = encoder_ckpt['input_layer.0.weight']
      # 		encoder_ckpt['input_layer.0.weight'] = altered_input_layer
      # 	return encoder_ckpt
      # else:
      print('Loading encoders weights from resnet34!')
      encoder_ckpt = torch.load(model_paths['resnet34'])
      # Transfer the RGB input of the resnet34 network to the first 3 input channels of pSp's encoder
      if self.opts.input_nc != 3:
        shape = encoder_ckpt['conv1.weight'].shape
        altered_input_layer = torch.randn(shape[0], self.opts.input_nc, shape[2], shape[3], dtype=torch.float32)
        altered_input_layer[:, :3, :, :] = encoder_ckpt['conv1.weight']
        encoder_ckpt['conv1.weight'] = altered_input_layer
      mapped_encoder_ckpt = dict(encoder_ckpt)
      for p, v in encoder_ckpt.items():
        for original_name, psp_name in RESNET_MAPPING.items():
          if original_name in p:
            mapped_encoder_ckpt[p.replace(original_name, psp_name)] = v
            mapped_encoder_ckpt.pop(p)
      return encoder_ckpt

    def forward(
        self,
        x,
        labels=None,
        resize=True,
        latent_mask=None,
        input_code=False,
        randomize_noise=True,
        noise=None,
        inject_latent=None,
        return_latents=False,
        alpha=None,
        truncation=1,
    ):

        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if self.opts.learn_in_w:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
                # else:
                # codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = (
                            alpha * inject_latent[:, i]
                            + (1 - alpha) * codes[:, i]
                        )
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        if noise is not None:

            codes = codes + noise

        input_is_latent = not input_code
        images, result_latent = self.decoder(
            [codes],
            labels=labels,
            input_is_latent=input_is_latent,
            randomize_noise=randomize_noise,
            return_latents=return_latents,
            truncation=truncation,
            truncation_latent=self.decoder.mean_latent(4096),
        )

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def set_opts(self, opts):
        # if opts.c_dim is not None:
        #     opts.c_dim = 0
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if "latent_avg" in ckpt:
            self.latent_avg = ckpt["latent_avg"].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None
