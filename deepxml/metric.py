# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
from functools import partial

def mlm_metric(y_true,y_pred):
	acc = accuracy_score(y_true, y_pred)
	return acc

def get_precision(y_true,y_pred,classes,top=5):
	mlb = MultiLabelBinarizer(classes=classes,sparse_output=True)
	mlb.fit(y_true)
	if not isinstance(y_true, csr_matrix):
		y_true = mlb.transform(y_true)
	y_pred = mlb.transform(y_pred[:,:top])
	return y_pred.multiply(y_true).sum() / (top * y_true.shape[0])

get_p_1 = partial(get_precision, top=1)
get_p_3 = partial(get_precision, top=3)
get_p_5 = partial(get_precision, top=5)

def get_ndcg(y_true, y_pred, classes, top=5):
	mlb = MultiLabelBinarizer(classes=classes,sparse_output=True)
	mlb.fit(y_true)
	if not isinstance(y_true, csr_matrix):
		y_true = mlb.transform(y_true)
	log = 1.0 / np.log2(np.arange(top) + 2)
	dcg = np.zeros((y_true.shape[0], 1))
	for i in range(top):
		p = mlb.transform(y_pred[:, i: i + 1])
		dcg += p.multiply(y_true).sum(axis=-1) * log[i]
	return np.average(dcg / log.cumsum()[np.minimum(y_true.sum(axis=-1), top) - 1])

get_n_3 = partial(get_ndcg, top=3)
get_n_5 = partial(get_ndcg, top=5)

def label2list(label):
	outputs = [[] for _ in range(label.shape[0])]
	x,y = np.where(label==1)
	for xx,yy in zip(x,y):
		outputs[xx].append(yy)
	return outputs

def xml_metric(y_true, y_score, classes, prt_sample_weight=True):
	sample_weight = np.sum(y_true, axis=1)
	sample_weight[sample_weight > 0] = 1
	if prt_sample_weight:
		print('sample_weight', np.sum(sample_weight), sample_weight.shape)
	y_true = label2list(y_true)
	y_pred = np.argsort(-y_score, axis=1)
	p1, p3, p5 = get_p_1(y_true, y_pred, classes), get_p_3(y_true, y_pred, classes), get_p_5(y_true, y_pred, classes)
	n3, n5 = get_n_3(y_true, y_pred, classes), get_n_5(y_true,y_pred, classes)
	return p1, p3, p5, n3, n5

def get_inv_propensity(train_y: csr_matrix, a=0.55, b=1.5):
    n, number = train_y.shape[0], np.asarray(train_y.sum(axis=0)).squeeze()
    c = (np.log(n) - 1) * ((b + 1) ** a)
    return 1.0 + c * (number + b) ** (-a)

def get_psp(targets, prediction, classes, inv_w, mlb, top=5):
	if not isinstance(targets, csr_matrix):
		targets = mlb.transform(targets)
	prediction = mlb.transform(prediction[:, :top]).multiply(inv_w)
	num = prediction.multiply(targets).sum()
	t, den = csr_matrix(targets.multiply(inv_w)), 0
	for i in range(t.shape[0]):
		den += np.sum(np.sort(t.getrow(i).data)[-top:])
	return num / den

get_psp_1 = partial(get_psp, top=1)
get_psp_3 = partial(get_psp, top=3)
get_psp_5 = partial(get_psp, top=5)

def xml_metric_psp(y_true, y_score, classes, train_labels):
	y_true = label2list(y_true)
	y_pred = np.argsort(-y_score, axis=1)
	mlb = MultiLabelBinarizer(classes=classes,sparse_output=True)
	mlb.fit(y_true)
	inv_w = get_inv_propensity(mlb.transform(train_labels), 0.55, 1.5)
	psp1, psp3, psp5 = get_psp_1(y_true, y_pred, classes, inv_w, mlb), get_psp_3(y_true, y_pred, classes, inv_w, mlb), get_psp_5(y_true, y_pred, classes, inv_w, mlb)
	return psp1, psp3, psp5