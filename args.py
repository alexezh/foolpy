class Args:
    def __init__(self):
        self.epochs = 10

        # embedding size, 200 default
        self.emsize = 8;
        self.nhid = 128;
        self.seq_length = 16

        self.nhead = 8;
        # number of neurons per layer
        self.kernel_size = 3;
        # number of layers
        self.nlayers = 2;
        # small model
        self.dropout = 0.3;
        self.seed = 42;
        self.model = "Transformer"
        self.batch_size = 64;
        self.lr = 1
        # sequence length
        self.clip = 0.25
        self.log_interval = 500
        self.dry_run = False
        self.save = "model.pt"
        self.relsFile = "rels.pt"
        self.onnx_export = False
        self.temperature = 1.0
        self.cuda = False;
        self.mps = True;

args = Args()