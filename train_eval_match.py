
import numpy as np
import torch
import time
from tensorboardX import SummaryWriter
from Utils.utils import classifiction_metric
from tqdm import tqdm


def train(epoch_num, model, train_dataloader, dev_dataloader, optimizer, criterion,
          label_list, out_model_file, log_dir, print_step, device):
    print('log path:', log_dir + '/' + time.strftime('%H_%M_%S', time.gmtime()))
    print('model path:', out_model_file)
    isPrint = True  # output
    writer = SummaryWriter(log_dir=log_dir + '/' + time.strftime('%H_%M_%S', time.gmtime()))

    global_step = 0
    best_dev_f1 =  0
    all_acc, all_f1 = np.array([], dtype=float), np.array([], dtype=float)
    all_dev_acc, all_dev_f1 = np.array([], dtype=float), np.array([], dtype=float)

    for epoch in tqdm(range(int(epoch_num))):
        print('---------------- Epoch: {} -----------------'.format(epoch+1))
        epoch_loss = 0
        train_steps = 0
        last_improve = 0
        flag = False

        all_preds = np.array([], dtype=int)
        all_labels = np.array([], dtype=int)

        for idx, batch in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            logits = model(batch['address1'],batch['address2'])


            '''loss'''
            loss = criterion(logits, batch['label'])
            if isPrint:
                print("cls_loss={}".format(loss))
                isPrint = False

            labels = batch['label'].detach().cpu().numpy()
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)

            loss.backward()
            optimizer.step()
            global_step += 1

            epoch_loss += loss.item()
            train_steps += 1

            all_preds = np.append(all_preds, preds)
            all_labels = np.append(all_labels, labels)

            if global_step % print_step == 0:
                train_loss = epoch_loss / train_steps
                train_acc, train_report = classifiction_metric(
                    all_preds, all_labels, label_list)

                dev_loss, dev_acc, dev_report = evaluate(
                    model, dev_dataloader, criterion, label_list, device, is_test=False)
                c = global_step // print_step

                writer.add_scalar("loss/train", train_loss, c)
                writer.add_scalar("loss/dev", dev_loss, c)

                writer.add_scalar("acc/train", train_acc, c)
                writer.add_scalar("acc/dev", dev_acc, c)
                print("acc/train", train_acc, c)
                print("acc/dev", dev_acc, c)

                for label in label_list:
                    writer.add_scalar(label + ":" + "f1/train",
                                      train_report[label]['f1-score'], c)
                    writer.add_scalar(label + ":" + "f1/dev",
                                      dev_report[label]['f1-score'], c)

                print_list = ['macro avg', 'weighted avg']
                for label in print_list:
                    writer.add_scalar(label + ":" + "f1/train",
                                      train_report[label]['f1-score'], c)
                    writer.add_scalar(label + ":" + "f1/dev",
                                      dev_report[label]['f1-score'], c)

                all_acc = np.append(all_acc, train_acc)
                all_f1 = np.append(all_f1, train_report[label]['f1-score'])
                all_dev_acc = np.append(all_dev_acc, dev_acc)
                all_dev_f1 = np.append(all_dev_f1, dev_report[label]['f1-score'])


                # f1
                #if dev_report['macro avg']['f1-score'] > best_dev_f1:
                best_dev_f1 = dev_report['macro avg']['f1-score']
                print('***********save model...*************')
                torch.save(model.state_dict(), out_model_file)
                last_improve = global_step
                model.train()

                if global_step - last_improve > 20000:
                    print('early stopping')
                    flag = True
                    break
        if flag == True:
            writer.close()
            break

    writer.close()


def evaluate(model, iterator, criterion, label_list, device, is_test=False):

    if is_test: print('start testing...')

    model.eval()
    epoch_loss = 0
    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)


    with torch.no_grad():
        for idx, batch in enumerate(iterator):
            logits = model(batch['address1'],batch['address2'] )

            '''loss'''
            loss = criterion(logits, batch['label'])
            labels = batch['label'].detach().cpu().numpy()
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            all_preds = np.append(all_preds, preds)
            all_labels = np.append(all_labels, labels)
            epoch_loss += loss.item()


    print("\ndev: epoch_loss={}".format(epoch_loss))
    acc, report = classifiction_metric(all_preds, all_labels, label_list)

    return epoch_loss / len(iterator), acc, report
