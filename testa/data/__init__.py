import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data.vqa_dataset import vqa_dataset
from data.pretrain_dataset import pretrain_dataset, pretrain_video_dataset
from data.video_dataset import VideoDataset, caption_video, caption_video_eval, CoinDataset, QuerYD_train, \
    QuerYD_retrieval_eval, MSRVTT_retrieval_train, MSRVTT_retrieval_eval, MSRVTT_qa, DiDeMo_retrieval_train, \
    DiDeMo_retrieval_eval, ActivityNet_qa, ActivityNet_retrieval_train, ActivityNet_retrieval_eval, \
    CondensedMovies_retrieval_train, CondensedMovies_retrieval_eval, ivqa, nextqa
from transform.randaugment import RandomAugment


def create_dataset(dataset, config, min_scale=0.5):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config['image_size'], scale=(min_scale, 1.0),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    transform_image_pseudo_video_train = transforms.Compose([                        
        transforms.RandomResizedCrop(config['image_size'], scale=(min_scale, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2,7,isPIL=True,augs=['Identity','Brightness','Sharpness',
                                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize
    ])    

    if dataset == 'pretrain':
        dataset = pretrain_dataset(config['train_file'], config['laion_path'], transform_train)
        return dataset
    elif dataset == 'pretrain_video':
        dataset = pretrain_video_dataset(config['train_file'], transform_image_pseudo_video_train,
            num_frm=config['num_frm_train'], max_img_size=config['image_size'], max_words=config['max_words'], 
            frm_sampling_strategy=config['frm_sampling_strategy']
        )
        return dataset

    elif dataset == 'caption_msrvtt':
        if 'input_segments' not in config:
            config['input_segments'] = False
        if 'input_asr' not in config:
            config['inpur_asr'] = False

        train_dataset = caption_video(config['video_root'], config['ann_root'], num_frm=config['num_frm_train'],
                                      max_img_size=config['image_size'], frm_sampling_strategy='nlvl_uniform',
                                      split='train', prompt=config['prompt'], video_resize=config['video_resize'],
                                      max_words=config['max_length'], input_segments=config['input_segments'], input_asr=config['input_asr'],
                                      asr_drop=config.get('train_asr_drop',0.0), seg_drop=config.get('train_seg_drop', 0.0))
        val_dataset = caption_video_eval(config['video_root'], config['ann_root'], num_frm=config['num_frm_val'],
                                         max_img_size=config['image_size'], frm_sampling_strategy='nlvl_uniform',
                                         split='val', input_segments=config['input_segments'], input_asr=config['input_asr'],
                                         asr_drop=config.get('test_asr_drop',0.0), seg_drop=config.get('test_seg_drop', 0.0))
        test_split = 'test'
        if 'VATEX' in config['ann_root']:
            test_split = 'val' #TODO: find test data
        test_dataset = caption_video_eval(config['video_root'], config['ann_root'], num_frm=config['num_frm_test'],
                                          max_img_size=config['image_size'], frm_sampling_strategy='nlvl_uniform',
                                          split=test_split, input_segments=config['input_segments'], input_asr=config['input_asr'],
                                          asr_drop=config.get('test_asr_drop',0.0), seg_drop=config.get('test_seg_drop', 0.0))
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'coin':
        train_dataset = CoinDataset(config['video_root'], config['ann_root'], num_frm=config['num_frm_train'],
                                    max_img_size=config['image_size'], frm_sampling_strategy='all',
                                    split='train', sliding_window=config['sliding_window'],
                                    sliding_window_size=config['sliding_window_size'], max_video_len=config['max_video_len'])
        val_dataset = CoinDataset(config['video_root'], config['ann_root'], num_frm=config['num_frm_test'],
                                  max_img_size=config['image_size'], frm_sampling_strategy='all',
                                  split='test', sliding_window=config['sliding_window'],
                                  sliding_window_size=config['sliding_window_size'], max_video_len=config['max_video_len'])
        test_dataset = val_dataset
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'retrieval_queryd':
        train_dataset = QuerYD_train(config['video_root'], config['ann_root'], num_frm=config['num_frm_train'],
                                     max_img_size=config['image_size'], frm_sampling_strategy='uniform',
                                     max_words=config['max_words'], split='train')
        val_dataset = QuerYD_retrieval_eval(config['video_root'], config['ann_root'], num_frm=config['num_frm_test'],
                                            max_img_size=config['image_size'], frm_sampling_strategy='uniform',
                                            max_words=config['max_words'], split='val')
        test_dataset = QuerYD_retrieval_eval(config['video_root'], config['ann_root'], num_frm=config['num_frm_test'],
                                             max_img_size=config['image_size'], frm_sampling_strategy='uniform',
                                             max_words=config['max_words'], split='test')
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'retrieval_msrvtt':
        train_dataset = MSRVTT_retrieval_train(config['video_root'], config['ann_root'], num_frm=config['num_frm_train'],
                                               max_img_size=config['image_size'], frm_sampling_strategy='uniform',
                                               max_words=config['max_words'], split='train')
        val_dataset = MSRVTT_retrieval_eval(config['video_root'], config['ann_root'], num_frm=config['num_frm_test'],
                                            max_img_size=config['image_size'], frm_sampling_strategy='uniform',
                                            max_words=config['max_words'], split='val')
        test_dataset = MSRVTT_retrieval_eval(config['video_root'], config['ann_root'], num_frm=config['num_frm_test'],
                                             max_img_size=config['image_size'], frm_sampling_strategy='uniform',
                                             max_words=config['max_words'], split='test')
        return train_dataset, val_dataset, test_dataset
    elif dataset == 'msrvtt_qa':
        train_dataset = MSRVTT_qa(config['video_root'], config['ann_root'], num_frm=config['num_frm_train'],
                                  max_img_size=config['image_size'], frm_sampling_strategy='uniform', split='train')
        val_dataset = MSRVTT_qa(config['video_root'], config['ann_root'], num_frm=config['num_frm_test'],
                                max_img_size=config['image_size'], frm_sampling_strategy='uniform', split='val')
        test_dataset = MSRVTT_qa(config['video_root'], config['ann_root'], num_frm=config['num_frm_test'],
                                 max_img_size=config['image_size'], frm_sampling_strategy='uniform', split='test')
        return train_dataset, val_dataset, test_dataset
    elif dataset == 'retrieval_didemo':
        train_dataset = DiDeMo_retrieval_train(config['video_root'], config['ann_root'],
                                               num_frm=config['num_frm_train'],
                                               max_img_size=config['image_size'], frm_sampling_strategy='uniform',
                                               max_words=config['max_words'], split='train')
        val_dataset = DiDeMo_retrieval_eval(config['video_root'], config['ann_root'], num_frm=config['num_frm_test'],
                                            max_img_size=config['image_size'], frm_sampling_strategy='uniform',
                                            max_words=config['max_words'], split='val')
        test_dataset = DiDeMo_retrieval_eval(config['video_root'], config['ann_root'], num_frm=config['num_frm_test'],
                                             max_img_size=config['image_size'], frm_sampling_strategy='uniform',
                                             max_words=config['max_words'], split='test')
        return train_dataset, val_dataset, test_dataset
    elif dataset == 'activitynet_qa':
        train_dataset = ActivityNet_qa(config['video_root'], config['ann_root'], num_frm=config['num_frm_train'],
                                       max_img_size=config['image_size'], frm_sampling_strategy='uniform',
                                       split='train')
        val_dataset = ActivityNet_qa(config['video_root'], config['ann_root'], num_frm=config['num_frm_test'],
                                     max_img_size=config['image_size'], frm_sampling_strategy='uniform', split='val')
        test_dataset = ActivityNet_qa(config['video_root'], config['ann_root'], num_frm=config['num_frm_test'],
                                      max_img_size=config['image_size'], frm_sampling_strategy='uniform', split='test')
        return train_dataset, val_dataset, test_dataset
    elif dataset == 'retrieval_activitynet':
        train_dataset = ActivityNet_retrieval_train(config['video_root'], config['ann_root'],
                                                    num_frm=config['num_frm_train'],
                                                    max_img_size=config['image_size'], frm_sampling_strategy='uniform',
                                                    split='train')
        val_dataset = ActivityNet_retrieval_eval(config['video_root'], config['ann_root'],
                                                 num_frm=config['num_frm_test'],
                                                 max_img_size=config['image_size'], frm_sampling_strategy='uniform',
                                                 split='val')
        test_dataset = ActivityNet_retrieval_eval(config['video_root'], config['ann_root'],
                                                  num_frm=config['num_frm_test'],
                                                  max_img_size=config['image_size'], frm_sampling_strategy='uniform',
                                                  split='val')
        return train_dataset, val_dataset, test_dataset
    elif dataset == 'retrieval_condensedmovies':
        train_dataset = CondensedMovies_retrieval_train(config['video_root'], config['ann_root'],
                                                        num_frm=config['num_frm_train'], max_img_size=config['image_size'],
                                                        frm_sampling_strategy='uniform', split='train')
        val_dataset = CondensedMovies_retrieval_eval(config['video_root'], config['ann_root'],
                                                     num_frm=config['num_frm_test'], max_img_size=config['image_size'],
                                                     frm_sampling_strategy='uniform', split='val')
        test_dataset = CondensedMovies_retrieval_eval(config['video_root'], config['ann_root'],
                                                      num_frm=config['num_frm_test'], max_img_size=config['image_size'],
                                                      frm_sampling_strategy='uniform', split='test')
        return train_dataset, val_dataset, test_dataset
    elif dataset == 'ivqa':
        train_dataset = ivqa(config['video_root'], config['ann_root'], num_frm=config['num_frm_train'],
                             max_img_size=config['image_size'], frm_sampling_strategy='uniform',
                             split='train')
        val_dataset = ivqa(config['video_root'], config['ann_root'], num_frm=config['num_frm_test'],
                           max_img_size=config['image_size'], frm_sampling_strategy='uniform', split='val')
        test_dataset = ivqa(config['video_root'], config['ann_root'], num_frm=config['num_frm_test'],
                            max_img_size=config['image_size'], frm_sampling_strategy='uniform', split='test')
        return train_dataset, val_dataset, test_dataset
    elif dataset == 'nextqa':
        train_dataset = nextqa(config['video_root'], config['ann_root'], num_frm=config['num_frm_train'],
                               max_img_size=config['image_size'], frm_sampling_strategy='nlvl_uniform',
                               split='train')
        val_dataset = nextqa(config['video_root'], config['ann_root'], num_frm=config['num_frm_test'],
                             max_img_size=config['image_size'], frm_sampling_strategy='nlvl_uniform', split='val')
        test_dataset = nextqa(config['video_root'], config['ann_root'], num_frm=config['num_frm_test'],
                              max_img_size=config['image_size'], frm_sampling_strategy='nlvl_uniform', split='test')
        return train_dataset, val_dataset, test_dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
