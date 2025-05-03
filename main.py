from wordencoder import trainWordEmbedding, loadWordEmbedding, testWord
from device import initDevice

initDevice()

# trainWordEmbedding();
loadWordEmbedding();
testWord("hello");
