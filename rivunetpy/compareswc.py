import os, argparse, json

from rivunetpy.utils.metrics import *
from rivunetpy.utils.io import loadswc, saveswc

def main(target, groundtruth, sigma=3):

    # If ground truth is present, use the ground truth to evaluate the reconstruction
    # The results will be written to the front of the swc file
    swc1 = loadswc(target)  # Load the ground truth
    swc2 = loadswc(groundtruth)  # Load the ground truth
    swc2[:, 5] = 1

    # PRECISION + RECALL
    (precision, recall, f1), (sd, ssd, pssd), prswc = precision_recall(
        swc1, swc2)  # Run precision&recall metrics
    fpath, _ = os.path.splitext(target)
    saveswc(fpath + '.node-compare.swc',
            prswc)  # Save the compare swc resulted from precision & recall

    # GAUSSIAN DISTANCE: Not used for now
    # gd1, gd2 = gaussian_distance(swc1, swc2, args.sigma)
    # print('G1 (FPR): %.2f\tG2 (FNR): %.2f' % (gd1.mean(), gd2.mean()))

    # CONECTIVITY ERRORS
    midx1, midx2 = connectivity_distance(swc1, swc2, sigma)
    # print('Precision:\tRecall:\tF1:\tC1\tC2')
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' %
          (precision, recall, f1, sd, ssd, pssd, len(midx1), len(midx2)))

    for i in midx1:
        swc1[i, 1] = 2
        swc1[i, 5] = 4
    saveswc(fpath + '.connect-err1.swc', swc1)

    for i in midx2:
        swc2[i, 1] = 2
        swc2[i, 5] = 4

    saveswc(fpath + '.connect-err2.swc', swc2)

    metrics = {}
    metrics['PRF'] = {'precision': precision, 'recall': recall, 'f1': f1}
    metrics['Distance'] = {'SD': sd, "SSD": ssd, "SSD%": pssd}
    # metrics['NetMetsGeometry'] = {'G1': gd1.mean(), 'G2': gd2.mean()}
    metrics['NetMetsConectivity'] = {'C1': len(midx1), 'C2': len(midx2)}
    # print('===================')
    # print()

    with open(fpath + '.metrics.json', 'w') as f:
        json.dump(metrics, f)