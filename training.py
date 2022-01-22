import torch
from tqdm import tqdm


def train_classifier(
    clf, optimizer_clf, train_loader, loss_criterion, epochs, device
):
    clf.to(device)
    total_classifier_loss = 0
    steps = 0
    pbar = tqdm(range(epochs))
    for _ in pbar:
        epoch_loss = 0
        epoch_batches = 0

        for data in train_loader:

            inputs, label, _ = data
            inputs, label = inputs.to(device), label.to(device)

            optimizer_clf.zero_grad()

            classifier_output = clf(inputs)
            classifier_loss = loss_criterion(classifier_output, label)
            classifier_loss.backward()
            optimizer_clf.step()
            total_classifier_loss += classifier_loss.item()
            epoch_loss += classifier_loss.item()
            epoch_batches += 1
            steps += 1

            pbar.set_description(
                f"Average Clf epoch loss: {epoch_loss/epoch_batches}"
            )

    print("Average Clf batch loss: ", total_classifier_loss / steps)

    return clf
