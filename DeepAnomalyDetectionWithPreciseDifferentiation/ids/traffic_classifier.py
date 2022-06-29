import sys
import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, fbeta_score, \
    precision_score, recall_score

from ids import utils
from ids.utils import get_device, time_since, create_tensor
from ids import utils


class BusClassifier:
    trained: bool
    num_epoch: int
    model_path: str
    model: torch.nn.Module

    def __init__(self, model: torch.nn.Module, criterion, optimizer,
                 model_path, num_epoch=300, train_log_output=sys.stdout, train_result_output=sys.stdout,
                 test_log_output=sys.stdout, test_result_output=sys.stdout):
        self.model = model
        self.num_epoch = num_epoch
        self.criterion = criterion
        self.optimizer = optimizer
        # train and test save
        self.model_path = model_path
        # 确定train的日志和结果保存位置
        self.train_log_output = train_log_output
        self.train_result_output = train_result_output
        self.test_log_output = test_log_output
        self.test_result_output = test_result_output

        self.trained = False
        self.binary = False
        # results
        self.train_result = {}
        self.test_result = {}
        self.class_labels = []
        self.test_log_path = None
        self.model_name = 'Default'

    def set_class_labels(self, class_labels):
        self.class_labels = class_labels
        if len(class_labels) > 2:
            self.binary = False
        else:
            self.binary = True

    def set_test_path(self, path):
        self.test_log_path = path

    def set_model_name(self, name):
        self.model_name = name

    def set_model_path(self, path):
        self.model_path = path

    def __repr__(self):
        info = f"""
Model:{self.model}
Model path:{self.model_path}
Epoch Num:{self.num_epoch}
Criterion:{self.criterion}
Optimizer:{self.optimizer}
"""
        return f"Bus Classifier({info})"

    # 训练模型
    def trainModel(self, train_loader):
        if self.binary:
            process = torch.nn.Sigmoid()
        # 定义总的损失
        total_train_loss = 0
        total_val_loss = 0
        start = time.time()
        case_count = 0
        train_avg_loss_all = []
        val_avg_loss_all = []
        val_acc_all = []
        train_acc_all = []
        train_epoch_time_cost_all = []
        epoch_train_num = 0
        epoch_val_num = 0
        for t in range(1, self.num_epoch + 1):
            epoch_train_num = 0
            epoch_val_num = 0
            epoch_train_loss = 0
            epoch_train_correct = 0
            epoch_val_loss = 0
            epoch_val_correct = 0
            loader_size = len(train_loader)
            for step, batch in enumerate(train_loader):
                if step <= 0.75 * loader_size:
                    self.model.train()
                    # 将数据从 train_loader 中读出来,一次读取的样本数是BATCH_SIZE个
                    train_input, train_label = batch
                    bz = len(train_label)  # batch size
                    epoch_train_num += bz
                    # 清除之前的梯度
                    self.model.zero_grad()
                    # 定义模型的输出
                    train_output = self.model(train_input)
                    if self.binary:
                        train_label = train_label.view(-1, 1)
                        pred = process(train_output).round()
                    else:
                        # 得到预测标签
                        pred = torch.argmax(train_output, dim=1)
                    # 比较输出与真实标签的loss
                    loss = self.criterion(train_output, train_label)
                    # 反向传播，更新权重
                    loss.backward()  # 梯度反传
                    self.optimizer.step()  # 更新参数
                    # 统计单个Batch结果，更新loss
                    epoch_train_loss += loss.item() * bz
                    epoch_train_correct += torch.sum(pred == train_label.data)
                else:
                    self.model.eval()
                    # 将数据从 train_loader 中读出来,一次读取的样本数是BATCH_SIZE个
                    train_input, train_label = batch
                    bz = len(train_label)  # batch size
                    epoch_val_num += bz
                    # 定义模型的输出
                    train_output = self.model(train_input)
                    if self.binary:
                        train_label = train_label.view(-1, 1)
                        pred = process(train_output).round()
                    else:
                        # 得到预测标签
                        pred = torch.argmax(train_output, dim=1)
                    # 比较输出与真实标签的loss
                    loss = self.criterion(train_output, train_label)
                    # 统计单个Batch结果，更新loss
                    epoch_val_loss += loss.item() * bz
                    epoch_val_correct += torch.sum(pred == train_label.data)
            # 统计单个Epoch结果
            case_count += epoch_train_num + epoch_val_num
            total_train_loss += epoch_train_loss
            total_val_loss += epoch_val_loss
            epoch_train_acc = epoch_train_correct.double().item() / epoch_train_num
            epoch_val_acc = epoch_val_correct.double().item() / epoch_val_num
            train_acc_all.append(epoch_train_acc)
            val_acc_all.append(epoch_val_acc)
            train_avg_loss_all.append(epoch_train_loss / epoch_train_num)
            val_avg_loss_all.append(epoch_val_loss / epoch_val_num)
            train_epoch_time_cost_all.append(time.time() - start)

            print(f'[{time_since(start)}] Epoch {t} ', end='', file=self.train_log_output)
            print('Epoch average loss={}'.format(epoch_train_loss / epoch_train_num), file=self.train_log_output)
            print('Epoch train acc={}%'.format(epoch_train_acc * 100), file=self.train_log_output)
            print('Epoch eval acc={}%'.format(epoch_val_acc * 100), file=self.train_log_output)
            print('Total average loss={}'.format(total_train_loss / case_count), flush=True, file=self.train_log_output)
        self.trained = True
        # 统计所有信息
        self.train_result['TimeCost'] = time_since(start)
        self.train_result['CaseNum'] = case_count
        self.train_result['TotalTrainLoss'] = total_train_loss
        self.train_result['TotalEvalLoss'] = total_val_loss
        self.train_result['TrainAccList'] = train_acc_all
        self.train_result['EvalAccList'] = val_acc_all
        self.train_result['TrainAvgLossList'] = train_avg_loss_all
        self.train_result['EvalAvgLossList'] = val_avg_loss_all
        self.train_result['TrainEpochTimeCostList'] = train_epoch_time_cost_all
        self.train_result['EpochTrainNum'] = epoch_train_num
        self.train_result['EpochEvalNum'] = epoch_val_num

        # 保存模型
        torch.save(self.model.state_dict(), self.model_path)
        # 保存训练结果
        print(self.train_result, flush=True, file=self.train_result_output)
        # 返回训练过程的所有统计信息
        return self.train_result

    def _to_binary(self, arr):
        """
        turn multiple classification result to binary classification result
        :param arr: result array like: [0, 1, 2, 3], and convert into: [0, 1, 1, 1]
        :return:
        """
        ret = [0] * len(arr)
        for i, item in enumerate(arr):
            if item == 0:
                ret[i] = 0
            else:
                ret[i] = 1
        return ret

    def _test_result_evaluate(self, outputs, targets):
        self.test_result = {}
        self.pred_lab = outputs
        self.true_lab = targets
        total_case = len(outputs)
        # 计算准确率
        acc = accuracy_score(targets, outputs)

        TN, FP, FN, TP = confusion_matrix(self._to_binary(targets), self._to_binary(outputs)).ravel()
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)

        percent = '%.2f' % (100 * acc)
        print(f'test set total case:{total_case}\n{percent}%', file=self.test_log_output)
        print(f'average loss:{self.test_total_loss / total_case}', file=self.test_log_output)
        self.test_result['Num'] = total_case
        self.test_result['TimeCost'] = self.test_time_cost
        self.test_result['AverageLoss'] = self.test_total_loss / total_case
        self.test_result['Acc'] = acc
        if not self.binary:
            precision, recall, f_score, _ = precision_recall_fscore_support(targets, outputs, beta=2.0,
                                                                            zero_division='warn')
            tpr, tnr, fpr, fnr = utils.get_multiclass_rate(outputs, targets)
            for i, label in enumerate(self.class_labels):
                self.test_result[f'{label}-Precision'] = precision[i]
                self.test_result[f'{label}-Recall'] = recall[i]
                self.test_result[f'{label}-F2Score'] = f_score[i]
                self.test_result[f'{label}-Tpr'] = tpr[i]
                self.test_result[f'{label}-Tnr'] = tnr[i]
                self.test_result[f'{label}-Fpr'] = fpr[i]
                self.test_result[f'{label}-Fnr'] = fnr[i]
        else:
            precision = precision_score(targets, outputs)
            recall = recall_score(targets, outputs)
            f_score = fbeta_score(targets, outputs, beta=2.0, zero_division='warn')
            self.test_result['Precision'] = precision
            self.test_result['Recall'] = recall
            self.test_result['FScore'] = f_score
            self.test_result['AUC'] = utils.cal_auc(outputs, targets)
            utils.draw_roc_curve(outputs, targets, 'Gru Roc Curve',
                                 f'{self.test_log_path}/Roc_Curve_{self.model_name}.png')

        self.test_result['Tpr'] = TPR
        self.test_result['Tnr'] = TNR
        self.test_result['Fpr'] = FPR
        self.test_result['Fnr'] = FNR
        print(self.test_result, file=self.test_result_output, flush=True)
        # draw_confusion_matrix(outputs[:], targets[:], self.class_labels, self.test_confusion_matrix_fig_name)
        return self.test_result

    def eval_model(self, test_loader):
        if not self.trained:
            self.model.load_state_dict(torch.load(self.model_path, map_location=get_device()))
        result = self._testModel(test_loader)
        # print([list(i) for i in result])
        return self._test_result_evaluate(*result)

    # 测试模型
    def _testModel(self, test_loader):
        if self.binary:
            process = torch.nn.Sigmoid()
        # 输出模型数据
        print(self, file=self.train_log_output)
        # 预测准确的个数
        total_loss = 0
        # 测试集的大小
        total_case = 0
        print('Evaluating trained model...', file=self.test_log_output)
        self.model.eval()
        start = time.time()
        outputs = np.array([])
        targets = np.array([])
        with torch.no_grad():
            for features, label in test_loader:
                # batch size
                bz = label.shape[0]
                total_case += bz
                # 定义相关的tensor
                inputs, target = features, label
                # 模型的输出
                output = self.model(inputs)
                if self.binary:
                    target = target.view(-1, 1)
                    pred = process(output).round()
                else:
                    # 得到预测标签
                    pred = torch.argmax(output, dim=1)
                # 比较输出与真实标签的loss
                loss = self.criterion(output, target)
                total_loss += loss.item() * bz
                outputs = np.hstack((outputs, pred.cpu().detach().numpy().flatten()))
                targets = np.hstack((targets, target.cpu().detach().numpy().flatten()))
        self.test_time_cost = time_since(start, detail=True)
        self.test_total_loss = total_loss
        print(f"Total evaluating time costs : {time_since(start, detail=True)}", file=self.test_log_output, flush=True)
        # 返回模型测试集的准确率
        return outputs, targets
