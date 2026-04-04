from matplotlib import pyplot as plt
from ..config import *


class Pipeline:
    def __init__(self, model, model_name: str, train_loader, val_loader, test_loader, class_names, device, epochs=10,
                 patience=5):
        self.model = model.to(device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'test_acc': 0,
            'misclassified': [],
            'confidence_stats': {}
        }

    def train(self, criterion, optimizer, scheduler=None):
        best_loss = float('inf')
        early_stop_counter = 0
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        for epoch in range(self.epochs):
            self.model.train()
            train_loss, correct = 0, 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()

            scheduler.step()

            self.model.eval()
            val_loss, val_correct = 0, 0
            all_confidences = []
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()

                    probabilities = torch.softmax(outputs, dim=1)
                    confidences = torch.max(probabilities, dim=1)[0]
                    all_confidences.extend(confidences.cpu().numpy())

            avg_train_loss = train_loss / len(self.train_loader)
            avg_val_loss = val_loss / len(self.val_loader)
            train_acc = correct / len(self.train_loader.dataset)
            val_acc = val_correct / len(self.val_loader.dataset)
            avg_confidence = np.mean(all_confidences) if all_confidences else 0

            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            if (epoch + 1) % 5 == 0:
                print(f"Результаты на {epoch + 1} эпохе:")
                print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4%}")
                print(f"   Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4%} | Conf: {avg_confidence:.2%}")

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.patience:
                    print(f"Ранняя остановка на {epoch + 1} эпохе | Best Loss: {best_loss:.4f}")
                    break

        torch.save(self.model.state_dict(), f'{MODELS_SAVE_PATH}/final_{self.model_name}.pth')

        self._collect_misclassified(self.test_loader)
        self.run_test()
        self.plot_errors()
        self.calculate_confidence_stats()

        return self.history

    def _collect_misclassified(self, loader, max_images=10):
        self.model.eval()
        collected = 0
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs_dev, lbls_dev = imgs.to(self.device), lbls.to(self.device)

                outputs = self.model(imgs_dev)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, preds = torch.max(probabilities, 1)

                mask = preds != lbls_dev

                if mask.any():
                    err_imgs = imgs_dev[mask]
                    err_preds = preds[mask]
                    err_labels = lbls_dev[mask]
                    err_confidences = confidences[mask]

                    for i in range(len(err_imgs)):
                        if collected < max_images:
                            self.history['misclassified'].append({
                                'image': err_imgs[i].cpu(),
                                'pred': err_preds[i].item(),
                                'label': err_labels[i].item(),
                                'confidence': err_confidences[i].item()
                            })
                            collected += 1
                        else:
                            break
                if collected >= max_images: break

    def calculate_confidence_stats(self):
        """Рассчитывает статистику уверенности для всех предсказаний"""
        self.model.eval()
        correct_confidences = []
        incorrect_confidences = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, preds = torch.max(probabilities, dim=1)

                correct_mask = (preds == labels)
                incorrect_mask = ~correct_mask

                correct_confidences.extend(confidences[correct_mask].cpu().numpy())
                incorrect_confidences.extend(confidences[incorrect_mask].cpu().numpy())

        self.history['confidence_stats'] = {
            'correct_mean': np.mean(correct_confidences) if correct_confidences else 0,
            'correct_std': np.std(correct_confidences) if correct_confidences else 0,
            'incorrect_mean': np.mean(incorrect_confidences) if incorrect_confidences else 0,
            'incorrect_std': np.std(incorrect_confidences) if incorrect_confidences else 0,
            'overall_mean': np.mean(correct_confidences + incorrect_confidences) if (
                    correct_confidences + incorrect_confidences) else 0
        }

        print(f"Статистика уверенности:")
        print(
            f"  Правильные предсказания: {self.history['confidence_stats']['correct_mean']:.2%} ± {self.history['confidence_stats']['correct_std']:.2%}")
        print(
            f"  Неправильные предсказания: {self.history['confidence_stats']['incorrect_mean']:.2%} ± {self.history['confidence_stats']['incorrect_std']:.2%}")
        print(f"  Общая уверенность: {self.history['confidence_stats']['overall_mean']:.2%}")

    def run_test(self):
        self.model.eval()
        test_correct = 0
        all_confidences = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, dim=1)
                test_correct += predicted.eq(labels).sum().item()
                all_confidences.extend(confidences.cpu().numpy())

        test_acc = test_correct / len(self.test_loader.dataset)
        avg_confidence = np.mean(all_confidences) if all_confidences else 0
        self.history['test_acc'] = test_acc

        print(f"Точность на тестовой выборке: {test_acc:.4%}")
        print(f"Средняя уверенность на тестовой выборке: {avg_confidence:.2%}")

    def plot_errors(self):
        errors = self.history['misclassified']
        if not errors:
            print("Ошибок не найдено.")
            return

        n = len(errors)
        plt.figure(figsize=(18, 4))
        for i, err in enumerate(errors):
            plt.subplot(1, n, i + 1)
            img = err['image'].permute(1, 2, 0)
            img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            plt.imshow(img.clamp(0, 1))
            confidence_text = f"Conf: {err['confidence']:.1%}" if 'confidence' in err else ""
            plt.title(
                f"True: {self.class_names[err['label']]}\nPred: {self.class_names[err['pred']]}\n{confidence_text}")
            plt.axis('off')
        plt.show()

    def plot_history(self):
        if len(self.history["train_loss"]) == 0:
            print("Сначала обучите модель!")
            return

        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.plot(self.history['train_acc'], label='Train Acc')
        plt.plot(self.history['val_acc'], label='Val Acc')
        plt.title('Accuracy history')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Loss history')
        plt.legend()

        if 'confidence_stats' in self.history and self.history['confidence_stats']:
            plt.subplot(1, 3, 3)
            conf_stats = self.history['confidence_stats']
            categories = ['Correct', 'Incorrect']
            means = [conf_stats['correct_mean'], conf_stats['incorrect_mean']]
            stds = [conf_stats['correct_std'], conf_stats['incorrect_std']]
            plt.bar(categories, means, yerr=stds, capsize=5)
            plt.title('Confidence Statistics')
            plt.ylabel('Confidence')

        plt.tight_layout()
        plt.show()
