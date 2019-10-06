from datasets.hmdb51_flow_faster import HMDB51OpticalFlowFaster


def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    assert opt.dataset in ['hmdb51_optical_flow_faster']

    if opt.dataset == 'hmdb51_optical_flow_faster':
        training_data = HMDB51OpticalFlowFaster(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform, epoch_multi=opt.epoch_multi)

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['hmdb51_optical_flow_faster']

    if opt.dataset == 'hmdb51_optical_flow_faster':
        validation_data = HMDB51OpticalFlowFaster(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['hmdb51_optical_flow_faster']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'hmdb51_optical_flow_faster':
        test_data = HMDB51OpticalFlowFaster(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)

    return test_data
