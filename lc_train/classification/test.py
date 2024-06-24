from tqdm import tqdm
import torch

def validation(test_loader, model, criterion, print_out = True, multi_class=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    total = 0
    correct = 0
    total_loss = 0
    # Iterate over batches
    with tqdm(test_loader, unit="validation", leave=False) as tepoch:
        for data, targets in tepoch:
            tepoch.set_description(f"test model ")

            # to 'cuda' or 'cpu'
            data, targets = data.to(device), targets.to(device)

            # get model output and loss and optim
            outputs = model(data.float())
            loss    = criterion(outputs, targets)

            # save all information
            if multi_class:
                _, predicted = torch.max(outputs.data, 1)
            else:
                predicted = (outputs > 0.5).float()
                    
            total       += targets.size(0)
            correct     += (predicted == targets).sum().item()
            total_loss  += loss.item()    

            tepoch.set_postfix(loss=loss.item(), accuracy=(predicted == targets).sum().item()/targets.size(0))
    
    # save train acc and loss
    current_accuracy = 100 * correct / total
    if print_out:
        print("Accuracy: ", round(current_accuracy,2), "Loss: ", round(total_loss / len(test_loader), 3))
    return current_accuracy, total_loss / len(test_loader)




