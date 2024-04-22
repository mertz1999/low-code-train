
# Low code train for pytorch community!

One of the exusted works in pytorch is writing codes for training part and printing result in best way also save them and import them easily. Therefore in this python package I implement a training pipeline for classification to make it easier for you.

    
## Intallation
First install this package

```bash
pip install git+https://github.com/mertz1999/low-code-train.git
```

## How to use it?

for using this package you need to define dataloader for classification and also you desire model, after that call this function

```python
from lc_train.classification import train

train.fit(trainloader, testloader, model, optimizer, criterion, epochs, resume=False, project='./')
```

consider that this training will save model in 2 format. First the best model accuracy will be save only state dict with the name of **best.pth** and also in each epoch it will save a lot of information in **last.pth** model file. 

- Last epoch number
- Last model state dict
- History of loss and accuracy (This will be discussed later)
- Last Optimizer state dict

If you want to find a model accuracy load your model and pass it to our **validation** function:

```python
from lc_train.classification import test

# If you want to use best.pth
model.load_state_dict(torch.load('best.pth'))

# If you want to use last.pth
loaded = torch.load('last.pth')
model.load_state_dict(loaded['model'])

accuracy, loss = test.validation(test_loader, model, criterion)
```

## Plotting
I said that the **last.pth** model has the history of the training (this information is used when **resume=True**) and now for plotting this use this simple block of code:

```python
loaded = torch.load('last.pth')
epoch_train_losses,epoch_train_accuracies,epoch_test_losses,epoch_test_accuracies = loaded['history']


plt.plot(epoch_train_losses, label='training')
plt.plot(epoch_test_losses, label='testing')
plt.title('Loss plot')
plt.legend()
plt.show()

plt.plot(epoch_train_accuracies, label='training')
plt.plot(epoch_test_accuracies, label='testing')
plt.title('Accuracy plot')
plt.legend()
plt.show()
```


## To Do In Future
- Plot confusion matrix (save them after training)
- Make a list of useful augmentation
- Make OOP project for using by anyone to edit training phase easily.

