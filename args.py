class Args:
    def __init__(self):
        # embedding size, 200 default
        self.emsize = 128;
        self.nhead = 8;
        # number of neurons per layer
        self.nhid = 128;
        # number of layers
        self.nlayers = 2;
        # small model
        self.dropout = 0.3;
        self.seed = 42;
        self.model = "Transformer"
        self.batch_size = 64;
        self.lr = 1
        self.epochs = 20
        # sequence length
        self.bptt = 48
        self.clip = 0.25
        self.log_interval = 100
        self.dry_run = False
        self.save = "model.pt"
        self.onnx_export = False
        self.temperature = 1.0
        self.cuda = False;
        self.mps = True;
