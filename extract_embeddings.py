from __future__ import division

import torch
import argparse

from onmt.ModelConstructor import make_embeddings, \
                            make_encoder, make_decoder

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-output_dir', default='.',
                    help="""Path to output the embeddings""")
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")


def write_embeddings(filename, dict, embeddings):
    with open(filename, 'w') as file:
        for i in range(len(embeddings)):
            outstr = str(dict.itos[i].encode("utf-8"))
            for j in range(len(embeddings[0])):
                outstr += " %5f" % (embeddings[i][j])
            file.write(outstr + "\n")


def main():
    opt = parser.parse_args()
    checkpoint = torch.load(opt.model)
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    model_opt = checkpoint['opt']
    src_dict = checkpoint['vocab'][0][1] 
    tgt_dict = checkpoint['vocab'][1][1]
    feature_dicts = []

    embeddings = make_embeddings(model_opt, src_dict, feature_dicts)
    encoder = make_encoder(model_opt, embeddings)

    embeddings = make_embeddings(model_opt, tgt_dict, feature_dicts,
                                 for_encoder=False)
    decoder = make_decoder(model_opt, embeddings)

    encoder_embeddings = encoder.embeddings.word_lut.weight.data.tolist()
    decoder_embeddings = decoder.embeddings.word_lut.weight.data.tolist()
    with open(r"enc_out.pkl", "wb") as output_file:
        torch.save(encoder.embeddings.word_lut.weight.data.cpu(),output_file)
    with open(r"dec_out.pkl", "wb") as output_file:
        torch.save(decoder.embeddings.word_lut.weight.data.cpu(),output_file)
            
#     print("Writing source embeddings")
#     write_embeddings(opt.output_dir + "/src_embeddings.txt", src_dict,
#                      encoder_embeddings)
# 
#     print("Writing target embeddings")
#     write_embeddings(opt.output_dir + "/tgt_embeddings.txt", tgt_dict,
#                      decoder_embeddings)

    print('... done.')
    print('Converting model...')


if __name__ == "__main__":
    main()
