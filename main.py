
# import v2v;
import args
from device import initDevice
import lstm2;


initDevice()



lstm2.complete("4 + 5 + 3", args, corpus)
lstm2.complete("4 + 5", args, corpus)
lstm2.complete("3 + 8 + 9", args, corpus)
lstm2.complete("5 + 3 + 8 + 9", args, corpus)
lstm2.complete("4 + 5 + 3 + 8 + 9", args, corpus)
lstm2.complete("a + 5 + 3", args, corpus)
lstm2.complete("3 + a + 3", args, corpus)



