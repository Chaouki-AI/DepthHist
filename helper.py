# This Code was written by Mohammed Chaouki ZIARA, 
# For Ph.D. Project @ RCAM Laboratory, Djilali Liabes University - Algeria 
# Below is the code for a functions used to build, train, evaluate model and for results visualization
# Contact: medchaoukiziara@gmail.com || me@chaouki.pro


from tools.loss import SILogLoss, Hist1D_loss, Hist2D_loss
from torch.utils.tensorboard import SummaryWriter
from tools.dataloader_root import DepthDataLoader
from Models.model import DepthHist 
from torchvision import transforms
from tools.evaluation import *
from datetime import datetime
from tqdm import tqdm
import numpy as np
import random
import torch
import os 


def count_parameters(model):
    """
    Computes the total number of trainable parameters in a given PyTorch model.
    
    The function iterates over all parameters of the model, filters out those
    that are not trainable (i.e., where requires_grad is False), and sums up 
    the number of elements in each trainable parameter tensor.

    Args:
        model (torch.nn.Module): The PyTorch model whose parameters need to be counted.

    Returns:
        int: The total number of trainable parameters in the model.
    """

    return sum(
        p.numel()  # Get the total number of elements in the parameter tensor
        for p in model.parameters()  # Iterate over all model parameters
        if p.requires_grad  # Consider only parameters that require gradient updates
    )


def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Parameters:
    seed (int): The seed to use for all random number generators.

    Notes:
    - This is important for reproducibility, as it ensures that the model will
      always produce the same results given the same inputs.
    - Also useful for debugging, as it will allow you to isolate any issues
      that are due to randomness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # If using CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(args):
    """
    Loads a model, initializes it with pre-trained weights (if provided), 
    and prints the total number of trainable parameters.

    Steps:
    1. Clears unused CUDA memory to free up space.
    2. Builds the model using the DepthHist class.
    3. Loads pre-trained weights if a checkpoint path is provided.
    4. Prints a confirmation message when weights are loaded.
    5. Prints the total number of trainable parameters in the model.

    Args:
        args (Namespace): A set of arguments containing:
            - path_pth_model (str or None): Path to the pre-trained model checkpoint.
    
    Returns:
        torch.nn.Module: The initialized model, optionally loaded with pre-trained weights.
    """

    # Clear unused memory in the CUDA cache to free GPU space
    torch.cuda.empty_cache()

    # Build the model using the DepthHist class (assumes DepthHist has a build method)
    model = DepthHist.build(args)

    # Load pre-trained weights if a checkpoint path is provided
    if args.path_pth_model is not None:
        model.load_state_dict(torch.load(args.path_pth_model))  # Load saved weights
        print("\n\n\nThe load of weights is done\n\n\n")  # Confirmation message

    # Print the total number of trainable parameters in the model
    print(f"\n\n\nThe total number of trainable parameters is: {count_parameters(model)}\n\n\n")

    return model  # Return the loaded model

def data_extractor_kitti(args, train=True):
    """
    Function Role:
    -----------------
    This function extracts image and depth file paths from the KITTI dataset.
    It reads a text file that contains the dataset split (train or test), constructs full paths, 
    verifies file existence, and returns valid paths.

    Inputs:
    -----------------
    - args: An object containing necessary dataset paths:
        - args.train_txt: Path to the file listing training samples.
        - args.test_txt: Path to the file listing test samples.
        - args.images_path: Base directory for images.
        - args.depths_path: Base directory for depth maps.
    - train (bool): If True, loads training data; if False, loads test/validation data.

    Returns:
    -----------------
    - imgs (list): A list of valid image file paths.
    - deps (list): A list of corresponding depth file paths.
    """

    file = open(args.train_txt) if train else open(args.test_txt)  # Opens the dataset split file
    lines = file.readlines()  # Reads all file lines
    file.close()  # Closes the file after reading

    # Defining base paths for images and depth maps
    path_tr_img = f"{args.images_path}"
    path_tr_dep = f"{args.depths_path}train/"
    
    path_ts_img = f"{args.images_path}"
    path_ts_dep = f"{args.depths_path}val/"
    
    imgs = []  # List to store valid image file paths
    deps = []  # List to store valid depth file paths
    
    for i in lines:
        # Extracting image and depth file names from the dataset split file
        image, depth = i.strip().split(' ')[:-1]

        # Formatting the image path correctly
        image = f"{image.split('/')[1]}/{image}"

        # Checking if the image and depth files exist in the training dataset
        if os.path.isfile(path_tr_img + image) and os.path.isfile(path_tr_dep + depth):
            imgs.append(path_tr_img + image)
            deps.append(path_tr_dep + depth)

        # Checking if the image and depth files exist in the test/validation dataset
        elif os.path.isfile(path_ts_img + image) and os.path.isfile(path_ts_dep + depth):
            imgs.append(path_ts_img + image)
            deps.append(path_ts_dep + depth)

    return imgs, deps  # Returning lists of valid image and depth file paths

def data_extractor_nyu(args, train=True):
    """
    Extracts the image and depth file paths from the NVUv2 dataset split files
    and returns them as two separate lists.
    
    Parameters
    ----------
    args (argparse.Namespace): The parsed command line arguments
    train (bool): Whether to load the training or test dataset (default: True)
    
    Returns
    -------
    imgs (list): A list of valid image file paths
    deps (list): A list of corresponding depth file paths
    """

    file = open(args.train_txt) if train else open(args.test_txt)
    lines = file.readlines()
    file.close()    
    imgs = []
    deps = []
    
    if args.dataset == 'nyu':
        # Loop through the lines in the dataset split file
        for i in lines:
            image, depth = i.strip().split(' ')[:-1]
            image_file = args.images_path+image if train else  args.images_path+'/'+image
            depth_file = args.depths_path+depth if train else  args.depths_path+'/'+depth
            # Check if both the image and depth files exist
            if os.path.isfile(image_file) and os.path.isfile(depth_file):
                imgs.append(image_file)
                deps.append(depth_file)
            else:
                print(args.images_path+image , os.path.isfile(depth_file))
    else:
        # Loop through the lines in the dataset split file
        for i in lines:
            line = args.images_path+i[:-1] if '//' not in i else (args.images_path+i).replace('//', '/')[:-1]
            image = line.split(' ')[0]
            depth = args.images_path+line.split(' ')[-1]  
            # Check if both the image and depth files exist
            if(os.path.isfile(image) and os.path.isfile(depth)):
                imgs.append(image)
                deps.append(depth)
            else:
                print(f" {image} or  {depth} isn't here ")
            
    return imgs, deps

def load_images_deps(args, train=True):
    """
    Loads image and depth file paths based on the specified dataset.

    Depending on the dataset type defined in args, this function extracts 
    and returns lists of image and depth file paths for either training or testing.

    Parameters:
    args (Namespace): The parsed command line arguments containing dataset configuration.
    train (bool): Flag indicating whether to load training data (True) or test data (False).

    Returns:
    tuple: A tuple containing two lists - image file paths and corresponding depth file paths.
    """

    if args.dataset == 'kitti':
        # Extract image and depth paths for KITTI dataset
        return data_extractor_kitti(args=args, train=train)
    elif args.dataset == 'nyu' or args.dataset == 'sunrgbd':
        # Extract image and depth paths for NYU or SUNRGBD dataset
        return data_extractor_nyu(args=args, train=train)
    
def load_dl(args, imgs, depths, train=True, mode=None):
    """
    Function to load data from disk into a dataloader object.

    Parameters:
    args (Namespace): The parsed command line arguments containing dataset configuration.
    imgs (list): A list of image file paths.
    depths (list): A list of corresponding depth file paths.
    train (bool): Flag indicating whether to load training data (True) or test data (False).
    mode (str): A string indicating the mode of the dataloader (train, test, or online_eval).

    Returns:
    dataloader (DataLoader): A DataLoader object containing the loaded data.
    """
    if train:
        # Create a DataLoader object in training mode
        dataloader = DepthDataLoader(args, imgs, depths, mode='train').data
    else:
        if mode == 'test':
            # Create a DataLoader object in test mode
            dataloader = DepthDataLoader(args, imgs, depths, mode='test').data
        else:
            # Create a DataLoader object in online evaluation mode
            dataloader = DepthDataLoader(args, imgs, depths, mode='online_eval').data
    return dataloader

def load_optimizer_and_scheduler(args, model, N_imgs):
    """
    Initializes the optimizer and learning rate scheduler based on the specified arguments.

    Args:
        args (Namespace): The parsed command line arguments containing optimizer and scheduler configurations.
        model (torch.nn.Module): The model whose parameters are to be optimized.
        N_imgs (int): The total number of images in the dataset.

    Returns:
        tuple: A tuple containing the initialized optimizer and scheduler.
    """

    # Retrieve the specified optimizer class from PyTorch's optim module
    optimizer_name = args.optimizer
    optimizer_class = getattr(torch.optim, optimizer_name)

    # Define parameter groups with different learning rates
    params = [{"params": model.get_1x_lr_params(), "lr": args.lr / 10},
              {"params": model.get_10x_lr_params(), "lr": args.lr}]

    # Calculate steps per epoch based on whether all images are used or a subset
    if args.all_images:
        steps_per_epoch = int(N_imgs / args.bs) if N_imgs % args.bs == 0 else int(N_imgs / args.bs) + 1
        print(f'iterations = {N_imgs / args.bs}')
    else:
        steps_per_epoch = int(args.Nb_imgs / args.bs)

    # Initialize the optimizer with appropriate settings
    if optimizer_name == 'AdamW' or optimizer_name == 'RMSprop':
        optimizer = optimizer_class(params, weight_decay=args.wd, lr=args.lr)
    else:
        optimizer = optimizer_class(params, weight_decay=args.wd, lr=args.lr, momentum=0.9, nesterov=True)

    # Initialize the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
        cycle_momentum=True, three_phase=False, base_momentum=0.85, max_momentum=0.95, last_epoch=-1,
        div_factor=args.epochs * 1000, anneal_strategy='linear', final_div_factor=args.epochs
    )

    return optimizer, scheduler

def load_losses(args):
    """
    Load the specified loss functions from the tools/loss.py module.

    Args:
        args (Namespace): The parsed command line arguments containing loss function configurations.

    Returns:
        tuple: A tuple containing the initialized loss functions.
    """
    # Load the loss functions from the tools/loss.py module
    Silog_loss = SILogLoss(args)
    OneDLoss = Hist1D_loss(args)
    TwoDLoss = Hist2D_loss(args)

    # Return the loss functions as a tuple
    return Silog_loss, TwoDLoss, OneDLoss

def pick_n_elements(args, list1, list2):
    """
    Pick a subset of images from two lists.

    If args.all_images is False, this function will randomly select a subset of
    elements from the two input lists. The number of elements to select is
    specified by args.Nb_imgs. If args.all_images is True, the function will
    return the original lists unchanged.

    Args:
        args (Namespace): The parsed command line arguments containing the
            configuration for the subset selection.
        list1 (list): The first list from which to select elements.
        list2 (list): The second list from which to select elements.

    Returns:
        tuple: A tuple containing the two lists with the selected elements.
    """
    if not args.all_images:
        # Ensure both lists have the same length
        assert len(list1) == len(list2), "Both lists must have the same length"
        
        # Generate a list of indices
        indices = list(range(len(list1)))
        
        # Randomly sample N indices from the list of indices
        selected_indices = random.sample(indices, args.Nb_imgs)
        
        # Use the selected indices to pick elements from both lists
        picked_list1 = [list1[i] for i in selected_indices]
        picked_list2 = [list2[i] for i in selected_indices]
        return picked_list1, picked_list2
    else :
        # If all images are to be used, return the original lists unchanged
        return list1, list2

def load_devices(args):
    """
    Determines and returns the devices for training and testing based on the availability
    of CUDA and the user's preference for GPU usage.

    Args:
        args (Namespace): A namespace containing the device configuration, specifically
                          'gpu_tr' and 'gpu_ts' flags indicating whether to use GPU
                          for training and testing respectively.

    Returns:
        tuple: A tuple containing the training device and testing device.
    """
    if torch.cuda.is_available():
        if args.gpu_tr and args.gpu_ts:
            # Use GPU for both training and testing
            print("\nGPU Will Be used For training and For Testing \n")
            device_tr = torch.device("cuda")
            device_ts = torch.device("cuda")
        elif args.gpu_tr and not args.gpu_ts:
            # Use GPU for training and CPU for testing
            print("\nGPU Will Be used For training and CPU For testing\n")
            device_tr = torch.device("cuda")
            device_ts = torch.device("cpu")
        elif args.gpu_ts and not args.gpu_tr:
            # Use CPU for training and GPU for testing
            print("\nCPU Will Be used For training and GPU For testing\n")
            device_tr = torch.device("cpu")
            device_ts = torch.device("cuda")
    else:
        # Use CPU for both training and testing if CUDA is not available
        print("\nCPU Will Be used For training and CPU For testing\n")
        device_tr = torch.device("cpu")
        device_ts = torch.device("cpu")
    
    return device_tr, device_ts

def trainer(args, model, dataloader, optimizer, scheduler, epoch, device, writer):
    """
    Train the model on a dataset for one epoch.

    Args:
        args (Namespace): The parsed command line arguments containing the
            configuration for the training.
        model (nn.Module): The model to be trained.
        dataloader (DataLoader): The data loader containing the training data.
        optimizer (Optimizer): The optimizer to be used for training.
        scheduler (Scheduler): The learning rate scheduler to be used for training.
        epoch (int): The current epoch number.
        device (torch.device): The device to be used for training.
        writer (SummaryWriter): The tensorboard writer for logging metrics.

    Returns:
        tuple: A tuple containing the trained model, optimizer, scheduler, and writer.
    """
    # Set the model to training mode
    model.train()

    # Load Losses Functions
    Si_loss, TwoDLoss, OneDLoss = load_losses(args)

    # Move the model to the specified device
    model = model.to(device)

    # Create a tqdm progress bar to track the training loop
    progress_bar = tqdm(enumerate(dataloader), desc=f"Epoch: {epoch + 1}/{args.epochs}. Loop: Train", total=len(dataloader))

    # Iterate over the mini-batches in the data loader
    for i, batch in progress_bar:
        # Get the input image and depth ground truth from the batch
        img = batch['image'].to(device)
        depth = batch['depth'].to(device)

        # Forward pass
        if args.simple:
            # If the model is simple, only predict the depth
            pred = model(img)
            histo = TwoDLoss.estimator(pred, TwoDLoss.__centers__(depth))
        else:
            # If the model is not simple, predict both the depth and histogram
            pred, histo = model(img)

        # Get the mask for pixels with valid depth values
        mask_min = depth > args.min_depth

        # Compute the loss
        l_dense = Si_loss(pred, depth, mask=mask_min.to(torch.bool), interpolate=True)

        # Compute the one-dimensional loss
        OD_Loss = OneDLoss(depth, pred, mask=mask_min.to(torch.bool), interpolate=True) if args.scale_hist != 0.0 else l_dense * 0
        # Compute the two-dimensional loss
        TD_Loss = TwoDLoss(depth, histo, mask=mask_min.to(torch.bool), interpolate=True) if args.scale_joint != 0.0 else l_dense * 0

        # Compute the total loss
        loss = args.scale_silog * l_dense + OD_Loss * args.scale_hist + TD_Loss * args.scale_joint  

        # Backpropagate the loss
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Update the learning rate scheduler
        scheduler.step()

        # Update the tqdm description with the current value of l_chamfer
        progress_bar.set_description(
            f"Epoch: {epoch + 1}/{args.epochs}. Loop: Train. SI_Loss: {l_dense.item():.4f} OD_Loss: {OD_Loss.item():.4f} TD_Loss: {TD_Loss.item():.4f}")

        # Log the loss and other metrics to tensorboard
        if i % 1 == 0:
            iteration_number = epoch * len(dataloader) + i
            writer.add_scalar('Loss/Train', loss.item(), iteration_number)
            writer.add_scalar('Loss/SILoss', l_dense.item(), iteration_number)
            writer.add_scalar("Loss/2DLoss", TD_Loss.item(), iteration_number)
            writer.add_scalar("Loss/1DLoss", OD_Loss.item(), iteration_number)
            writer.add_scalar('PAR/LR', scheduler.get_last_lr()[0], iteration_number)

    # Return the trained model, optimizer, scheduler, and writer
    return model, optimizer, scheduler, writer

class InverseNormalize(transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        # The inverse of the normalization is: (input * std) + mean
        inv_std = 1 / std
        inv_mean = -mean / std
        super().__init__(inv_mean.tolist(), inv_std.tolist())

def __evaluator__(args, model, test_loader, epoch, range, device, writer):
    """
    Evaluate the model on a dataset for a given range.

    Args:
        args (Namespace): The parsed command line arguments containing the
            configuration for the evaluation.
        model (nn.Module): The model to be evaluated.
        test_loader (DataLoader): The data loader containing the test data.
        epoch (int): The current epoch number.
        range (float): The maximum depth range for evaluation.
        device (torch.device): The device to be used for evaluation.
        writer (SummaryWriter): The tensorboard writer for logging metrics.

    Returns:
        tuple: A tuple containing the evaluation metrics, validation Si loss,
               and the tensorboard writer.
    """
    # Load Si loss function
    Si_loss, _, _ = load_losses(args)
    
    # Set the model to evaluation mode
    model.eval()

    # Move the model to the specified device
    model = model.to(device)
    
    with torch.no_grad():
        # Initialize running averages for Si loss and metrics
        val_si = RunningAverage()
        metrics = RunningAverageDict()
        
        # Iterate over the batches in the test data loader
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{args.epochs} Validation for {int(range)} Meters")):
            # Get the input image and depth ground truth from the batch
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)

            # Process depth dimensions
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)

            # Forward pass
            if not args.simple:
                pred, _ = model(img)  # Predict both depth and histogram
            else:
                pred = model(img)  # Only predict depth

            # Create a mask for valid depth values
            mask = depth > args.min_depth

            # Compute Si loss
            l_dense = Si_loss(pred, depth, mask=mask.to(torch.bool), interpolate=True)
            val_si.append(l_dense.item())

            # Resize predictions to match depth shape
            pred = torch.nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)

            # Post-process predictions
            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth] = args.min_depth
            pred[pred > range] = range
            pred[np.isinf(pred)] = range
            pred[np.isnan(pred)] = args.min_depth

            # Get ground truth depth and valid mask
            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth > args.min_depth, gt_depth < range)

            # Determine evaluation mask based on cropping strategy
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)
            if args.crop == "garg":
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                          int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
            elif args.crop == "eigen":
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                              int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu' or args.dataset == 'sunrgbd':
                    eval_mask[45:471, 41:601] = 1

            # Update valid mask and compute errors
            valid_mask = np.logical_and(valid_mask, eval_mask)
            metrics.update(compute_errors(gt_depth[valid_mask], pred[valid_mask]))

        # Format metric values
        values = {key: float(f"{value:.5f}") for key, value in metrics.get_value().items()}

        # Log metrics to tensorboard
        writer.add_scalar(f'Metrics {int(range)} Meters/rmse', values['rmse'], epoch + 1)
        writer.add_scalar(f'Metrics {int(range)} Meters/sq_rel', values['sq_rel'], epoch + 1)
        writer.add_scalar(f'Metrics {int(range)} Meters/rmse_log', values['rmse_log'], epoch + 1)
        writer.add_scalar(f'Metrics {int(range)} Meters/acc1', values['a1'], epoch + 1)
        writer.add_scalar(f'Metrics {int(range)} Meters/acc2', values['a2'], epoch + 1)
        writer.add_scalar(f'Metrics {int(range)} Meters/acc3', values['a3'], epoch + 1)
        writer.add_scalar(f'Metrics {int(range)} Meters/abs_rel', values['abs_rel'], epoch + 1)

        return metrics.get_value(), val_si, writer

def evaluator(args, model, valid_dl, epoch, device, writer, name):
    """
    Evaluate the model on the validation set for the specified range.
    
    Args:
        args (Namespace): The parsed command line arguments containing model and experiment configurations.
        model (nn.Module): The model to be evaluated.
        valid_dl (DataLoader): The validation data loader.
        epoch (int): The current epoch number.
        device (torch.device): The device to be used for evaluation.
        writer (SummaryWriter): The tensorboard writer for logging metrics.
        name (str): The name of the experiment.

    Returns:
        tuple: A tuple containing the evaluated model, the filename of the saved metrics, the updated writer, and the metrics.
    """
    if args.dataset == 'kitti':
        # Evaluate for 80 meters and save on the txt file
        metrics_80, _ , writer = __evaluator__(args, model, valid_dl, epoch, range = 80., device = device, writer = writer)
        metrics_80 = {key: float(f"{value:.5f}") for key, value in metrics_80.items()}
        filename = save_metrics_to_file(args, metrics_80, epoch, 80., name=name)
        return model, filename, writer, metrics_80
    elif args.dataset == 'nyu' or args.dataset == 'sunrgbd':
        # Evaluate for 10 meters and save on the txt file
        metrics_10, _ , writer = __evaluator__(args, model, valid_dl, epoch, range = 10., device = device, writer = writer)
        metrics_10 = {key: float(f"{value:.5f}") for key, value in metrics_10.items()}
        filename = save_metrics_to_file(args, metrics_10, epoch, 10., name=name)
        return model, filename, writer, metrics_10

def save_metrics_to_file(args, metrics, epoch, range, name):
    """
    Save evaluation metrics to a file.

    Args:
        args (Namespace): The parsed command line arguments.
        metrics (dict): A dictionary containing the evaluation metrics.
        epoch (int): The current epoch number.
        range (float): The maximum depth range for evaluation.
        name (str): The name of the experiment.

    Returns:
        str: The filename where the metrics were saved.
    """
    # Construct the filename for saving metrics
    filename = f"{args.ckpt}{name}/metrics.txt"

    # Prepare additional information for the first epoch
    if epoch == 0:
        dets = '\n'.join([f"{arg} : {getattr(args, arg)}" for arg in vars(args)])
        explain = f"{dets}\n"
        sep = str('*' * 100)
        additional_info = f"{explain}{sep}\n\nMetrics for {name} \n\n"
        additional_info = f'{additional_info}Epoch:{epoch + 1} for range {int(range)}'
    else:
        # Prepare additional information for subsequent epochs
        additional_info = f"Epoch:{epoch + 1}/{args.epochs} - MaxDepth {int(range)}"

    # Open the file in append mode and write metrics
    with open(filename, 'a') as file:
        if additional_info:
            file.write(additional_info + "\n")
        # Write each metric to the file
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")
        # Add a separator line if range is 60
        if range == 60.:
            file.write(f"{str('-' * 100)}\n")
        file.write("\n")

    return filename

def save_ckpt(args, model, metrics, name, epoch):
    """
    Save the model checkpoint.

    Args:
        args (Namespace): The parsed command line arguments.
        model (nn.Module): The model to be saved.
        metrics (dict): A dictionary containing the evaluation metrics.
        name (str): The name of the experiment.
        epoch (int): The current epoch number.
    """
    # Construct the checkpoint filename using relevant metrics and epoch number
    checkpoint_filename = os.path.join(
        args.ckpt,
        f"{name}",
        f"epoch-{epoch+1}_abs_rel-{metrics['abs_rel']}_A1-{metrics['a1']}_best.pt"
    )
    
    # Save the model state dictionary to the specified file
    torch.save(model.state_dict(), checkpoint_filename)

def load_summary_writer(args, name):
    """
    Load the tensorboard summary writer for the specified experiment name.

    Args:
        args (Namespace): The parsed command line arguments containing experiment configurations.
        name (str): The name of the experiment.

    Returns:
        SummaryWriter: The tensorboard summary writer.
    """
    # Create the directory for storing tensorboard logs if it doesn't exist
    os.makedirs(f"{args.ckpt}{name}",  exist_ok=True)

    # Initialize the tensorboard summary writer
    writer = SummaryWriter(args.runs+name)

    return writer

def load_weights(args, model, path=None, device=None):
    """
    Load the pre-trained weights for a model.

    Args:
        args (Namespace): The parsed command line arguments containing experiment configurations.
        model (nn.Module): The model to load pre-trained weights for.
        path (str): The path to the pre-trained weights file.
        device (str): The device to use for loading the pre-trained weights.

    Returns:
        nn.Module: The model with pre-trained weights loaded.
    """
    # If no device is specified, load the device from the command line arguments
    if device is None:
        _, device = load_devices(args)
    else:
        # If a device is specified, convert it to a torch device
        device = torch.device(device)

    try:
        # Load the pre-trained weights from the specified file
        model.load_state_dict(torch.load(path, map_location=device))
    except Exception as e:
        # If there is an error loading the pre-trained weights, print the error
        print(e)

    return model


