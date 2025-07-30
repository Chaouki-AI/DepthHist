# This Code was written by Mohammed Chaouki ZIARA, 
# For Ph.D. Project @ RCAM Laboratory, Djilali Liabes University - Algeria 
# Below is the code for a functions used to build, train, evaluate model and for results visualization
# Contact: medchaoukiziara@gmail.com || chaouki.ziara@univ-sba.dz


from tools.loss import SILogLoss, Hist2D_loss, CenterLoss
from torch.utils.tensorboard import SummaryWriter
from Models.model import DepthHist
import matplotlib.pyplot as plt
from torchvision import transforms
from tools.dataloader_root import DepthDataLoader
from tools.evaluation import *
from tqdm import tqdm
import numpy as np
import random
import torch
import os 


def load_model(args):
    torch.cuda.empty_cache()
    model = DepthHist.build(args)
    if args.path_pth_model != None : 
        model.load_state_dict(torch.load(args.path_pth_model))
        print("\nThe load of weights is done\n")
    
    print(f"\n\n\nThe total Number of trainable parameters will be : {count_parameters(model)}")
    print(f"The total Number of trainable parameters on encoder will be : {count_parameters(model.encoder)}")
    print(f"The total Number of trainable parameters on decoder will be : {count_parameters(model.decoder)}")
    print(f"The total Number of trainable parameters on histlayer will be : {count_parameters(model.Histogram)}\n\n\n")
    return model

def load_images_deps(args, train = True):
    if args.dataset == 'kitti':
        return data_extractor_kitti(args=args, train=train)
    else :
        return data_extractor_nyu(args=args, train=train)
 
def data_extractor_kitti(args, train = True):

    file  = open(args.train_txt) if train else open(args.test_txt)
    lines = file.readlines()
    file.close()
    path_tr_img = f"{args.images_path}"
    path_tr_dep = f"{args.depths_path}train/"
    
    path_ts_img = f"{args.images_path}"
    path_ts_dep = f"{args.depths_path}val/"
    
    
    imgs = []
    deps = []
    
    for i in lines :
        image, depth = i.strip().split(' ')[:-1]
        image = f"{image.split('/')[1]}/{image}" 
        if os.path.isfile(path_tr_img+image) and os.path.isfile(path_tr_dep+depth):
            imgs.append(path_tr_img+image)
            deps.append(path_tr_dep+depth)

        elif os.path.isfile(path_ts_img+image) and os.path.isfile(path_ts_dep+depth):
            imgs.append(path_ts_img+image)
            deps.append(path_ts_dep+depth)
        else : 
            pass 
            #print(f"{image} or {depth} doesn't exist")
            
    return imgs, deps

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

def load_dl(args, imgs, depths, train = True, mode = None):
    if train : 
        dataloader = DepthDataLoader(args, imgs, depths, mode = 'train').data
    else :
        if mode == 'test':
            dataloader = DepthDataLoader(args, imgs, depths, mode = 'test').data
        else :
            dataloader = DepthDataLoader(args, imgs, depths, mode = 'online_eval').data
    return dataloader

def load_optimizer_and_scheduler(args, model, N_imgs):
    optimizer_name = args.optimizer
    optimizer_class = getattr(torch.optim, optimizer_name)
    
    params = [{"params": model.get_1x_lr_params() , "lr": args.lr/10},
              {"params": model.get_10x_lr_params(), "lr": args.lr}]
    
    steps_per_epoch = int(N_imgs/args.bs) if N_imgs % args.bs == 0 else int(N_imgs/args.bs) + 1
    
    if optimizer_name == 'AdamW' or optimizer_name == 'RMSprop':  
        optimizer = optimizer_class(params, weight_decay=args.wd, lr=args.lr)
    else :
        optimizer = optimizer_class(params, weight_decay=args.wd, lr=args.lr, momentum=0.9, nesterov=True)

    if args.lr_pol == 'OCL' :
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, epochs=args.epochs, 
                                                        steps_per_epoch= steps_per_epoch,
                                                        cycle_momentum=True, three_phase = False,
                                                        base_momentum=0.85, max_momentum=0.95, last_epoch=-1,
                                                        div_factor = 100, anneal_strategy = 'linear', 
                                                        final_div_factor = args.epochs)
    else : 
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, create_lr_lambda(steps_per_epoch*args.epochs,args.lr))

    return optimizer, scheduler

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def create_lr_lambda(total_steps, max_lr):
    """
    Creates an LR lambda function for the 1-cycle policy:
    - Linear warm-up for the first 30% of total steps.
    - Cosine annealing for the remaining 70% of total steps.

    Args:
        total_steps (int): Total number of training steps (iterations).
        max_lr (float): Maximum learning rate.

    Returns:
        lr_lambda (function): Function to compute learning rate factor based on current step.
    """
    warmup_steps = int(0.3 * total_steps)  # 30% of total steps for warm-up
    min_lr = max_lr / 75  # Minimum learning rate during cosine annealing
    warmup_start_lr = max_lr / 25  # Starting LR for warm-up

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warm-up phase
            return warmup_start_lr + (max_lr - warmup_start_lr) * (current_step / warmup_steps)
        else:
            # Cosine annealing phase
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr + 0.5 * (max_lr - min_lr) * (1 + torch.cos(torch.tensor(progress * 3.14159)))

    return lr_lambda

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

def load_losses(args):
    Silog_loss = SILogLoss(args)
    Hist_loss = Hist2D_loss(args)
    Center_loss = CenterLoss(args)

    return Silog_loss, Hist_loss, Center_loss

def load_devices(args):
    if torch.cuda.is_available():
        if args.gpu_tr and args.gpu_ts :
            print("\nGPU Will Be used For training and For Testing \n")
            device_tr = torch.device("cuda")
            device_ts = torch.device("cuda")

        elif args.gpu_tr and not args.gpu_ts :
            print("\nGPU Will Be used For training and CPU For testing\n")
            device_tr = torch.device("cuda")
            device_ts = torch.device("cpu")

        elif args.gpu_ts and not args.gpu_tr :
            print("\nCPU Will Be used For training and GPU For testing\n")
            device_tr = torch.device("cpu")
            device_ts = torch.device("cuda")
    else :
        print("\nCPU Will Be used For training and CPU For testing\n")
        device_tr = torch.device("cpu")
        device_ts = torch.device("cpu")
    return device_tr, device_ts


def trainer(args, model, dataloader, optimizer, scheduler, epoch, device, writer):
    model.train()

    #Load Losses Functions
    Si_loss, Hist_loss, CentLoss = load_losses(args)
    CentLoss.to(device)
    #device, _ = get_device(args, epoch)
    accumulated_loss = 0
    model = model.to(device)
    progress_bar = tqdm(enumerate(dataloader), desc=f"Epoch: {epoch + 1}/{args.epochs}. Loop: Train", total=len(dataloader))
    for i, batch in progress_bar:
        img = batch['image'].to(device)
        depth = batch['depth'].to(device)
        #print(img.shape)
        pred, histogram, centers = model(img)

        mask_min = depth > args.min_depth
        
        SILogLoss  = Si_loss(pred, depth, mask=mask_min.to(torch.bool), interpolate=True)
        CentersLoss  = CentLoss(depth, pred, centers, mask=mask_min.to(torch.bool), interpolate = True)  if args.scale_center  != 0.0 else SILogLoss*0
        j_histo  = Hist_loss(depth, histogram, centers, model.Histogram.scales, mask=mask_min.to(torch.bool), interpolate = True) if args.scale_hist != 0.0 else SILogLoss*0

        
        loss = args.scale_silog * SILogLoss  + CentersLoss * args.scale_center + j_histo * args.scale_hist #+ (scale_penalty) * args.scale_silog
        
        if i % 3 == 0:
            iteration_number = epoch * len(dataloader) + i
            writer.add_scalar('Loss/Train', loss.item()            , iteration_number)
            writer.add_scalar('Loss/Silog', SILogLoss.item()       , iteration_number)
            writer.add_scalar('Loss/Hist', j_histo.item()          , iteration_number)
            writer.add_scalar('Loss/Center ', CentersLoss.item()   , iteration_number)
            writer.add_scalar('PAR/LR' , scheduler.get_last_lr()[0], iteration_number)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
        optimizer.step()
        scheduler.step()
        
        

        # Update the tqdm description with the current value of l_chamfer
        progress_bar.set_description(f"Epoch: {epoch + 1}/{args.epochs}. Loop: Train. Silog: {SILogLoss.item():.4f} Center: {CentersLoss.item():.4f} j_histo: {j_histo.item():.4f}")
            
        
    return model, optimizer, scheduler, writer 


def __evaluator__(args, model, test_loader, epoch, range , device, writer):
    Si_loss, _, _ = load_losses(args)
    model.eval()


    model = model.to(device)
    with torch.no_grad():
        val_si = RunningAverage()
        metrics = RunningAverageDict()
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{args.epochs} Validation for {int(range)} Meters")):
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            
            
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
 
            pred = model(img)[0]

            mask = depth > args.min_depth

            l_dense = Si_loss(pred, depth, mask=mask.to(torch.bool), interpolate=True)
            val_si.append(l_dense.item())

            pred = torch.nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)

            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth] = args.min_depth
            pred[pred > range] = range
            pred[np.isinf(pred)] = range
            pred[np.isnan(pred)] = args.min_depth

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth > args.min_depth, gt_depth < range)
            

            if args.dataset != 'diode':

                if args.crop == "garg":
                    #print('here')
                    gt_height, gt_width = gt_depth.shape
                    eval_mask = np.zeros(valid_mask.shape)
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.crop == "eigen":
                    gt_height, gt_width = gt_depth.shape
                    eval_mask = np.zeros(valid_mask.shape)
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                                  int(0.0359477 * gt_width) :int(0.96405229 * gt_width)] = 1
                    elif args.dataset == 'nyu' or args.dataset == 'sunrgbd':
                        eval_mask[45:471, 41:601] = 1
            else :
                eval_mask = np.ones(valid_mask.shape)
            valid_mask = np.logical_and(valid_mask, eval_mask)
            metrics.update(compute_errors(gt_depth[valid_mask], pred[valid_mask]))
        
        values = {key: float(f"{value:.5f}") for key, value in metrics.get_value().items()}
        writer.add_scalar(f'Metrics {int(range)} Meters/rmse'    , values['rmse'] , epoch+1)
        writer.add_scalar(f'Metrics {int(range)} Meters/sq_rel'  , values['sq_rel'] , epoch+1)
        writer.add_scalar(f'Metrics {int(range)} Meters/rmse_log', values['rmse_log'] , epoch+1)
        writer.add_scalar(f'Metrics {int(range)} Meters/acc1'    , values['a1'] , epoch+1)
        writer.add_scalar(f'Metrics {int(range)} Meters/acc2'    , values['a2'] , epoch+1)
        writer.add_scalar(f'Metrics {int(range)} Meters/acc3'    , values['a3'] , epoch+1)
        writer.add_scalar(f'Metrics {int(range)} Meters/abs_rel' , values['abs_rel'] , epoch+1)

        return metrics.get_value(), val_si, writer

def evaluator(args, model, valid_dl, epoch , device, writer, name):
    if args.dataset == 'kitti' :
        #evaluate for 80meters and save on the txt file  
        metrics_80, _ , writer = __evaluator__(args, model, valid_dl, epoch, range = 80., device = device, writer = writer)
        metrics_80 = {key: float(f"{value:.5f}") for key, value in metrics_80.items()}
        filename = save_metrics_to_file(args, metrics_80, epoch, 80., name=name)

        #evaluate for 60meters and save on the txt file  
        # metrics_60, _, writer = __evaluator__(args, model, valid_dl, epoch, range = 60., device = device, writer = writer)
        # metrics_60 = {key: float(f"{value:.5f}") for key, value in metrics_60.items()}
        # filename = save_metrics_to_file(args, metrics_60, epoch, 60., name=name)
        return model, filename, writer, metrics_80
    
    elif args.dataset == 'nyu' or args.dataset == 'sunrgbd' :
        #evaluate for 10meters and save on the txt file
        metrics_10, _ , writer = __evaluator__(args, model, valid_dl, epoch, range = 10., device = device, writer = writer)
        metrics_10 = {key: float(f"{value:.5f}") for key, value in metrics_10.items()}
        filename = save_metrics_to_file(args, metrics_10, epoch, 10., name=name)
        return model, filename, writer, metrics_10
    

def save_metrics_to_file(args, metrics, epoch, range, name):
    filename = f"{args.ckpt}{name}/metrics.txt"
    if epoch == 0 :
        dets = '\n'.join([f"{arg} : {getattr(args, arg)}" for arg in vars(args)])
        explain = f"{dets}\n"
        sep = str('*'*100)
        additional_info = f"{explain}{sep}\n\nMetrics for {name} \n\n"
        additional_info = f'{additional_info}Epoch:{epoch+1} for range {int(range)}'
    else : 
        additional_info =  f"Epoch:{epoch+1}/{args.epochs} - MaxDepth {int(range)}"

    with open(filename, 'a') as file:
        if additional_info:
            file.write(additional_info + "\n")
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")
        if range == 60. :
            file.write(f"{str('-'*100)}\n") 
        file.write("\n") 
    return filename

def save_ckpt(args, model, metrics, name, epoch):
    
    torch.save(model.state_dict(), os.path.join(args.ckpt, f"{name}", f"epoch-{epoch+1}_abs_rel-{metrics['abs_rel']}_A1-{metrics['a1']}_best.pt"))

def load_summary_writer(args, name):
    os.makedirs(f"{args.ckpt}{name}",  exist_ok=True)
    writer = SummaryWriter(args.runs+name)
    return writer

def load_weights(args, model, path=None, device = 'cpu'):
    if device is None : 
        _, device = load_devices(args)
    else : 
        device = torch.device(device)
    try :
        model.load_state_dict(torch.load(path, map_location = device))
    except Exception as e :
        print(e)
    return model


