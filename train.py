import torch
import wandb
import argparse
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from os.path import join

# Set CUDA architecture list and float32 matmul precision high
from sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()
torch.set_float32_matmul_precision('high')

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module_gcommands import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model_gcommands import WhisperGuidedScoreModel


def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups

def load_pretrained_diffusion_model(
    model: WhisperGuidedScoreModel, 
    pretrained_ckpt_path: str,
    freeze_pretrained: bool = False
) -> WhisperGuidedScoreModel:
    """
    Load pretrained weights into a new model with additional parameters.
    
    Args:
        model: The new model with additional parameters
        pretrained_ckpt_path: Path to the pretrained checkpoint
        freeze_pretrained: Whether to freeze the loaded pretrained parameters
    
    Returns:
        Model with loaded pretrained weights
    """
    print(f"Loading pretrained weights from: {pretrained_ckpt_path}")
    
    # Load checkpoint
    checkpoint = torch.load(pretrained_ckpt_path, map_location='cpu', weights_only=False)
    
    if 'state_dict' in checkpoint:
        pretrained_state_dict = checkpoint['state_dict']
    else:
        pretrained_state_dict = checkpoint
    
    # Get current model parameters
    model_dict = model.state_dict()
    
    # Filter pretrained weights to match current model
    pretrained_filtered = {}
    new_parameters = []
    
    for name, param in model_dict.items():
        if name in pretrained_state_dict:
            pretrained_param = pretrained_state_dict[name]
            if param.shape == pretrained_param.shape:
                pretrained_filtered[name] = pretrained_param
            else:
                print(f"Shape mismatch for {name}: keeping random initialization")
                new_parameters.append(name)
        else:
            # This is a new parameter (like self.class_processor, self.class_film_generator)
            new_parameters.append(name)
    
    # Load the filtered weights
    model.load_state_dict(pretrained_filtered, strict=False)
    
    print(f"Loaded {len(pretrained_filtered)} pretrained parameters")
    print(f"New/randomly initialized parameters: {len(new_parameters)}")
    
    # Optionally freeze pretrained parameters
    if freeze_pretrained:
        for name, param in model.named_parameters():
            if name in pretrained_filtered:
                param.requires_grad = False
    
    return model

if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
          parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")
          parser_.add_argument("--nolog", action='store_true', help="Turn off logging.")
          parser_.add_argument("--wandb_name", type=str, default=None, help="Name for wandb logger. If not set, a random name is generated.")
          parser_.add_argument("--ckpt", type=str, default=None, help="Resume training from checkpoint.")
          parser_.add_argument("--pretrained_ckpt", type=str, default="/home/gilad/diffusion_EM/ASR_diffusion/original_sgmse_from_pretrained/logs/sgmse_original_weights_reverb.ckpt",
                                help="Path to an older checkpoint whose overlapping params should initialise the new model.")
          parser_.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs.")
          parser_.add_argument("--save_ckpt_interval", type=int, default=50000, help="Save checkpoint interval.")
          parser_.add_argument("--test", action='store_true', help="Run in test mode - load checkpoint and run validation.")
          # Add new arguments for Whisper-guided model
          parser_.add_argument("--model_mode", type=str, default="regular", choices=["regular", "guided", "distilled"],
                              help="Model mode: regular, guided (Whisper), or distilled (teacher-student)")
          parser_.add_argument("--whisper_name", type=str, default="base", 
                              help="Whisper model size (tiny, base, small, medium, large)")
          parser_.add_argument("--guidance_scale", type=float, default=1.0,
                              help="Scale factor for Whisper guidance")
          parser_.add_argument("--distillation_weight", type=float, default=1.0,
                              help="Weight for distillation loss (only for distilled mode)")
          parser_.add_argument("--whisper_lang", type=str, default="en",
                              help="Language for Whisper decoding")

          
     temp_args, _ = base_parser.parse_known_args()

     # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     sde_class = SDERegistry.get_by_name(temp_args.sde)
     trainer_parser = parser.add_argument_group("Trainer", description="Lightning Trainer")
     trainer_parser.add_argument("--accelerator", type=str, default="gpu", help="Supports passing different accelerator types.")
     trainer_parser.add_argument("--devices", default="auto", help="How many gpus to use.")
     trainer_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients.")
     trainer_parser.add_argument("--max_epochs", type=int, default=-1, help="Number of epochs to train.")
     
     WhisperGuidedScoreModel.add_argparse_args(
          parser.add_argument_group("ScoreModel", description=WhisperGuidedScoreModel.__name__))
     sde_class.add_argparse_args(
          parser.add_argument_group("SDE", description=sde_class.__name__))
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     # Parse args and separate into groups
     args = parser.parse_args()
     arg_groups = get_argparse_groups(parser)

     # Initialize logger, trainer, model, datamodule
     model = WhisperGuidedScoreModel(
          backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
          # Add model mode and guidance parameters
          model_mode=args.model_mode,
          whisper_name=args.whisper_name,
          guidance_scale=args.guidance_scale,
          distillation_weight=args.distillation_weight,
          whisper_lang=args.whisper_lang,
          **{
               **vars(arg_groups['ScoreModel']),
               **vars(arg_groups['SDE']),
               **vars(arg_groups['Backbone']),
               **vars(arg_groups['DataModule'])
          }
     )

     if hasattr(args, 'pretrained_ckpt') and args.pretrained_ckpt:
        model = load_pretrained_diffusion_model(
            model, 
            args.pretrained_ckpt,
            freeze_pretrained=getattr(args, 'freeze_pretrained', False)
        )

     # Set up logger configuration
     if args.nolog:
          logger = None
     else:
          # Add model mode to WandB project name
          wandb_project = f"sgmse_{args.model_mode}"
          if args.model_mode in ["guided", "distilled"]:
               wandb_project += f"_{args.whisper_name}"
               
          logger = WandbLogger(project=wandb_project, log_model=False, save_dir="logs", name=args.wandb_name)
          logger.experiment.log_code(".")

     # Set up callbacks for logger
     if logger != None:
          callbacks = [ModelCheckpoint(dirpath=join(args.log_dir, str(logger.version)), save_last=True, 
               filename='{epoch}-last')]
          callbacks += [ModelCheckpoint(dirpath=join(args.log_dir, f'{str(logger.version)}-{args.wandb_name}'),
               filename='{step}', save_top_k=-1, every_n_train_steps=args.save_ckpt_interval)]
          if args.num_eval_files:
               checkpoint_callback_pesq = ModelCheckpoint(dirpath=join(args.log_dir, str(logger.version)), 
                    save_top_k=5, monitor="wer_enhanced", mode="min", filename='{epoch}-{wer_enhanced:.3f}')
               callbacks += [checkpoint_callback_pesq]
               
               # Add teacher model metrics tracking for distilled mode
               if args.model_mode == "distilled":
                    checkpoint_callback_teacher_pesq = ModelCheckpoint(
                         dirpath=join(args.log_dir, str(logger.version)), 
                         save_top_k=5, monitor="wer_enhanced", mode="min", 
                         filename='{epoch}-teacher-{teacher_pesq:.2f}'
                    )
                    callbacks.append(checkpoint_callback_teacher_pesq)
     else:
          callbacks = None

     # Initialize the Trainer and the DataModule
     trainer = pl.Trainer(
          **vars(arg_groups['Trainer']),
          strategy="ddp_find_unused_parameters_true", logger=logger,
          log_every_n_steps=10, num_sanity_val_steps=0,
          callbacks=callbacks
     )

     # Train model
     if args.test:
          if args.ckpt is None:
               raise ValueError("Checkpoint path must be provided in test mode")
          # Ensure data module is set up for validation
          trainer.validate(model, ckpt_path=args.ckpt)
     else:
          trainer.fit(model, ckpt_path=args.ckpt)