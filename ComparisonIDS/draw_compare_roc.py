from matplotlib import pyplot as plt
from sklearn import metrics

plt.rcParams['font.sans-serif'] = "Times New Roman"

if __name__ == '__main__':
    stan = ([0., 0.17590505, 1.], [0., 0.92766987, 1.])
    secon = ([0, 0.12871116, 1.], [0., 0.88438966, 1.])
    s7 = ([0., 0.00211009, 1.], [0., 0.98674673, 1.])
    s5 = ([0., 0.09344297, 1.], [0., 0.97372153, 1.])
    ax = plt.gca()
    display = metrics.RocCurveDisplay(fpr=stan[0], tpr=stan[1], roc_auc=metrics.auc(*stan),
                                      estimator_name="Stan Roc Curve")
    display_secon = metrics.RocCurveDisplay(fpr=secon[0], tpr=secon[1], roc_auc=metrics.auc(*secon),
                                            estimator_name="Qiao Roc Curve")
    display_s7 = metrics.RocCurveDisplay(fpr=s7[0], tpr=s7[1], roc_auc=metrics.auc(*s7),
                                         estimator_name="Gru Roc Curve")
    display.plot(ax, color='#C07A92')
    display_secon.plot(ax, color='#DFC286')
    display_s7.plot(ax, color='#608595')
    plt.show()
