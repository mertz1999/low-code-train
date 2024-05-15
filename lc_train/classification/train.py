from tqdm import tqdm
import torch
from .test import validation
import os

def fit(train_loder, test_loader, model, optimizer, criterion, epochs, resume=False, project='./'):
    model.train()
    print('\n(info) Start training')
    # load pre-trained model for resume
    if resume:
        try:
            # load model and it's different parts
            loaded = torch.load(os.path.join(project,'last.pth'))
            model.load_state_dict(loaded['model'])
            optimizer.load_state_dict(loaded['optimizer'])
            start_epoch = loaded['epoch']
            epoch_train_losses,epoch_train_accuracies,epoch_test_losses,epoch_test_accuracies = loaded['history']
            print(f"(info) model is loaded from {os.path.join(project, 'last.pth')}")
        except:
            raise Exception('(error) problem in loading model')
    else:
        start_epoch            = 0
        epoch_train_losses     = []
        epoch_train_accuracies = []
        epoch_test_losses      = []
        epoch_test_accuracies  = []


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(start_epoch, epochs):
        model.train()
        total = 0
        correct = 0
        total_loss = 0
        # Iterate over batches
        with tqdm(train_loder, unit="batch", leave=False) as tepoch:
            for data, targets in tepoch:
                tepoch.set_description(f"Epoch {epoch+1} train")

                # to 'cuda' or 'cpu'
                data, targets = data.to(device), targets.to(device)

                optimizer.zero_grad()

                # get model output and loss and optim
                outputs = model(data)
                loss    = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # save all information
                _, predicted = torch.max(outputs.data, 1)
                total       += targets.size(0)
                correct     += (predicted == targets).sum().item()
                total_loss  += loss.item()
                

                tepoch.set_postfix(loss=loss.item(), accuracy=(predicted == targets).sum().item()/targets.size(0))
        
        # save train acc and loss
        current_accuracy = 100 * correct / total
        epoch_train_losses.append(total_loss / len(train_loder))
        epoch_train_accuracies.append(current_accuracy)

        # make validation
        acc, loss = validation(test_loader, model, criterion)
        epoch_test_losses.append(loss)
        epoch_test_accuracies.append(acc)

        # save model
        torch.save({
            'epoch'    : epoch,
            'model'    : model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history'  : (epoch_train_losses,epoch_train_accuracies,epoch_test_losses,epoch_test_accuracies)
            }, os.path.join(project,'last.pth'))

        if acc > max(epoch_test_accuracies[:-1]):
            torch.save(model.state_dict(), os.path.join(project,'best.pth'))
            print('(info) Best model is saved!')

        
        print(f'Epoch {epoch+1} : Valid Accuracy {round(acc,3)}\t Valid Loss {round(loss,3)}\t Train Accu {round(current_accuracy,3)}\t Train Loss {round(total_loss / len(train_loder),3)}')


