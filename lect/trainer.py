
import os
import torch


# Triplet trainer
class TripletTrainer:
    def __init__(self,
                model,
                device,
                train_loader,
                valid_loader,
                criterion,
                optimizer,
                scheduler,
                epochs,
                model_name = 'my-model',
                save_interval = False,
                if_dry_run = False,
                log_dir="logs"):
        super(TripletTrainer, self).__init__()
        # train and valid
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        # log and save
        self.save_interval = save_interval
        self.name = model_name
        self.dry_run = if_dry_run

    def train(self):
        self.model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            anchor, positive, negative = [sample for sample in batch]

            self.optimizer.zero_grad()

            # Forward pass
            anchor_embeddings = self.model(anchor["img"].to(self.device))
            positive_embeddings = self.model(positive["img"].to(self.device))
            negative_embeddings = self.model(negative["img"].to(self.device))

            anchor_loc = [coord.to(self.device) for coord in anchor["coords"]]
            positive_loc = [coord.to(self.device) for coord in positive["coords"]]
            negative_loc = [coord.to(self.device) for coord in negative["coords"]]

            # Compute triplet loss
            loss = self.criterion(anchor_embeddings, positive_embeddings, negative_embeddings, anchor_loc, positive_loc, negative_loc)

            # Backward and optimize
            loss.backward()
            self.optimizer.step()

            # loss
            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)

        # average loss within one epoch
        return avg_loss

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    # validation
    def valid(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            correct = 0
            total = 0

            for batch_idx, batch in enumerate(self.valid_loader):
                anchor, positive, negative = [sample for sample in batch]

                anchor_embeddings = self.model(anchor["img"].to(self.device))
                positive_embeddings = self.model(positive["img"].to(self.device))
                negative_embeddings = self.model(negative["img"].to(self.device))

                anchor_loc = [coord.to(self.device) for coord in anchor["coords"]]
                positive_loc = [coord.to(self.device) for coord in positive["coords"]]
                negative_loc = [coord.to(self.device) for coord in negative["coords"]]

                # Compute triplet loss
                loss = self.criterion(anchor_embeddings, positive_embeddings, negative_embeddings, anchor_loc, positive_loc, negative_loc)
                # loss
                running_loss += loss.item()

                # Calculate distances
                positive_distance = self.calc_euclidean(anchor_embeddings, positive_embeddings)
                negative_distance = self.calc_euclidean(anchor_embeddings, negative_embeddings)

                # Count correct predictions (lower distance for positive pairs, higher distance for negative pairs)
                correct += torch.sum(positive_distance < negative_distance).item()
                total += anchor_embeddings.size(0)

            avg_loss = running_loss/len(self.valid_loader)
            acc = correct / total

        return avg_loss, acc


    def run(self):

        for epoch in range(self.epochs):

            train_loss = self.train()
            valid_loss, acc = self.valid()
            self.scheduler.step()
            print(f"Epoch {epoch + 1}/{self.epochs}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f} Valid Accuracy: {acc * 100:.2f}%")

            # for training test
            if self.dry_run:
                break

            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f} Valid Accuracy: {acc * 100:.2f}%")

            if self.save_interval and (epoch+1) % self.save_interval == 0:

                if not os.path.exists(f"saved_models/{self.name}"):
                    os.makedirs(f"saved_models/{self.name}")
                model_path = f"saved_models/{self.name}/{self.name}-{epoch+1}-ckpt.pth"

                torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': train_loss,
                        }, model_path)


# Triplet trainer
class One2ManyTrainer:
    def __init__(self,
                model,
                device,
                train_loader,
                valid_loader,
                criterion,
                optimizer,
                scheduler,
                epochs,
                model_name = 'my-model',
                save_interval = False,
                if_dry_run = False,
                log_dir="logs"):
        super(One2ManyTrainer, self).__init__()
        # train and valid
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs

        # log and save
        self.writer = SummaryWriter(log_dir=f'{log_dir}/{model_name}_tensorboard/')
        self.logger = logging.basicConfig(filename=f'{log_dir}/{model_name}.log', encoding='utf-8', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.save_interval = save_interval
        self.name = model_name
        self.dry_run = if_dry_run

    def train(self):
        self.model.train()
        running_loss = 0.0

        for _, batch in enumerate(self.train_loader):
            anc, pos, neg = [sample for sample in batch]

            self.optimizer.zero_grad()

            # Forward pass
            ancEm = self.model(anc["img"].to(self.device))
            posEm = self.model(pos["img"].to(self.device))
            negEm = torch.empty(ancEm.size(0), neg['img'].size(1), ancEm.size(1)).to(self.device)
            for i in range(neg['img'].size(1)):
                negEm[:, i, :] = self.model(neg["img"][:, i, :, :, :].to(self.device))

            ancLoc = [coord.to(self.device) for coord in anc["coords"]]
            posLoc = [coord.to(self.device) for coord in pos["coords"]]
            negLoc = []
            for coords in neg["coords"]:
                thisCoords = []
                for coord in coords:
                    thisCoords.append(coord.to(self.device))
                negLoc.append(thisCoords)
            # negLoc = [[coord.to(self.device)] for coords in neg["coords"] for coord in coords]

            # Compute triplet loss
            loss = self.criterion(ancEm, posEm, negEm, ancLoc, posLoc, negLoc)

            # Backward and optimize
            loss.backward()
            self.optimizer.step()

            # loss
            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)

        # average loss within one epoch
        return avg_loss

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    # validation
    def valid(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            correct = 0
            total = 0

            for _, batch in enumerate(self.valid_loader):
                anc, pos, neg = [sample for sample in batch]

                ancEm = self.model(anc["img"].to(self.device))
                posEm = self.model(pos["img"].to(self.device))
                negEm = torch.empty(ancEm.size(0), neg['img'].size(1), ancEm.size(1)).to(self.device)
                for i in range(neg['img'].size(1)):
                    negEm[:, i, :] = self.model(neg["img"][:, i, :, :, :].to(self.device))

                ancLoc = [coord.to(self.device) for coord in anc["coords"]]
                posLoc = [coord.to(self.device) for coord in pos["coords"]]
                negLoc = []
                for coords in neg["coords"]:
                    thisCoords = []
                    for coord in coords:
                        thisCoords.append(coord.to(self.device))
                    negLoc.append(thisCoords)

                # Compute triplet loss
                loss = self.criterion(ancEm, posEm, negEm, ancLoc, posLoc, negLoc)
                running_loss += loss.item()

                # Calculate distances
                posDist = self.calc_euclidean(ancEm, posEm)
                for i in range(neg['img'].size(1)):
                    negDist = self.calc_euclidean(ancEm, negEm[:, i, :])
                    correct += torch.sum(posDist < negDist).item()
                total += ancEm.size(0) * neg['img'].size(1)

            avg_loss = running_loss/len(self.valid_loader)
            acc = correct / total

        return avg_loss, acc


    def run(self):

        for epoch in range(self.epochs):

            train_loss = self.train()
            valid_loss, acc = self.valid()
            self.scheduler.step()
            print(f"Epoch {epoch + 1}/{self.epochs}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f} Valid Accuracy: {acc * 100:.2f}%")

            # for training test
            if self.dry_run:
                break

            self.writer.add_scalar(f'Loss/train', train_loss, epoch)
            self.writer.add_scalar(f'Loss/Valid', valid_loss, epoch)
            self.writer.add_scalar(f'Accuracy/Valid', acc, epoch)

            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f} Valid Accuracy: {acc * 100:.2f}%")

            if self.save_interval and (epoch+1) % self.save_interval == 0:

                if not os.path.exists(f"saved_models/{self.name}"):
                    os.makedirs(f"saved_models/{self.name}")
                model_path = f"saved_models/{self.name}/{self.name}-{epoch+1}-ckpt.pth"

                torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': train_loss,
                        }, model_path)

        self.writer.close()


# ImgLabel trainer
class ImgLabelTrainer:
    def __init__(self,
                model,
                device,
                train_loader,
                valid_loader,
                criterion,
                optimizer,
                scheduler,
                epochs,
                mode='classification',  # or embedding
                model_name = 'my-model',
                save_interval = False,
                if_dry_run = False,
                log_dir="logs"):
        super(ImgLabelTrainer, self).__init__()
        # train and valid
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.mode = mode

        # log and save
        self.writer = SummaryWriter(log_dir=f'{log_dir}/{model_name}_tensorboard/')
        self.logger = logging.basicConfig(filename=f'{log_dir}/{model_name}.log', encoding='utf-8', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.save_interval = save_interval
        self.name = model_name
        self.dry_run = if_dry_run

    def train(self, epoch=-1):
        self.model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # Forward pass
            image, label, coord = [sample for sample in batch]
            output = self.model(image.to(self.device))

            # Compute loss
            if self.mode == 'classification':
                loss = self.criterion(output, label.to(self.device))
            elif self.mode == 'embedding':
                coord = [thisCoord.to(self.device) for thisCoord in coord]
                loss = self.criterion(self.model, image.to(self.device), label.to(self.device), epoch, loc=coord)
            # Backward and optimize
            loss.backward()
            self.optimizer.step()

            # loss
            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)

        # average loss within one epoch
        return avg_loss

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    # validation
    def valid(self, epoch=-1):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            correct = 0
            total = 0

            for batch_idx, batch in enumerate(self.valid_loader):

                image, label, coord = [sample for sample in batch]
                output = self.model(image.to(self.device))

                # Compute loss
                if isinstance(self.criterion, type(torch.nn.CrossEntropyLoss())):
                    loss = self.criterion(output, label.to(self.device))
                else:
                    # TODO: Fix here, snnl
                    loss = self.criterion(self.model, image.to(self.device), label.to(self.device), epoch, loc=coord)
                # loss
                running_loss += loss.item()

                # count correct
                if self.mode == 'classification':
                    correct_predictions = sum(torch.argmax(output, dim=1)==label.to(self.device)).item()
                    correct += correct_predictions
                    total += len(image)
                elif self.mode == 'embedding':
                    # ancEm = output[label==0,:]
                    ancEm, posEm, negEm = [], [], []
                    for l in torch.unique(label):
                        thisAncPos = output[label==l]
                        thisNeg = output[label!=l]
                        for i in range(thisAncPos.size(0)):
                            thisAncEm = thisAncPos[i]
                            for j in range(thisAncPos.size(0)):
                                if i != j:
                                    thisPosEm = thisAncPos[j]
                                    for k in range(thisNeg.size(0)):
                                        thisNegEm = thisNeg[k]
                                        ancEm.append(thisAncEm)
                                        posEm.append(thisPosEm)
                                        negEm.append(thisNegEm)
                    ancEm = torch.stack(ancEm)
                    posEm = torch.stack(posEm)
                    negEm = torch.stack(negEm)
                    posDist = self.calc_euclidean(ancEm, posEm)
                    negDist = self.calc_euclidean(ancEm, negEm)
                    # Count correct predictions (lower distance for positive pairs, higher distance for negative pairs)
                    correct += torch.sum(posDist < negDist).item()
                    total += ancEm.size(0)

            avg_loss = running_loss/len(self.valid_loader)
            acc = correct / total

        return avg_loss, acc


    def run(self):

        for epoch in range(self.epochs):

            train_loss = self.train(epoch=epoch)
            valid_loss, acc = self.valid(epoch=epoch)
            self.scheduler.step()
            print(f"Epoch {epoch + 1}/{self.epochs}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f} Valid Accuracy: {acc * 100:.2f}%")

            # for training test
            if self.dry_run:
                break

            self.writer.add_scalar(f'Loss/train', train_loss, epoch)
            self.writer.add_scalar(f'Loss/Valid', valid_loss, epoch)
            self.writer.add_scalar(f'Accuracy/Valid', acc, epoch)

            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f} Valid Accuracy: {acc * 100:.2f}%")

            if self.save_interval and (epoch+1) % self.save_interval == 0:

                if not os.path.exists(f"saved_models/{self.name}"):
                    os.makedirs(f"saved_models/{self.name}")
                model_path = f"saved_models/{self.name}/{self.name}-{epoch+1}-ckpt.pth"

                torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': train_loss,
                        }, model_path)
        self.writer.close()

