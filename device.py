import torch

# import v2v;
import args

device = None

# code is based on
# https://github.com/pytorch/examples/blob/main/word_language_model/main.py

def initDevice():
  global device
  # Set the random seed manually for reproducibility.
  torch.manual_seed(args.args.seed)
  if torch.cuda.is_available():
      if not args.args.cuda:
          print("WARNING: You have a CUDA device, so you should probably run with --cuda.")
  if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
      if not args.args.mps:
          print("WARNING: You have mps device, to enable macOS GPU run with --mps.")

  use_mps = args.args.mps and torch.backends.mps.is_available()
  if args.args.cuda:
      device = torch.device("cuda")
  elif use_mps:
      device = torch.device("mps")
  else:
      device = torch.device("cpu")
