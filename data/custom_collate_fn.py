def custom_collate_fn(batch):
    """
    Custom collate that forces clean tensor creation.
    This solves the 'storage not resizable' error definitively.
    """
    images = []
    labels = []

    for image, label in batch:
        # Force completely new tensors with fresh storage
        image_clean = torch.tensor(image.detach().cpu().numpy(), dtype=torch.float32)
        label_clean = torch.tensor(label.detach().cpu().numpy(), dtype=torch.float32)

        images.append(image_clean)
        labels.append(label_clean)

    # Stack with error handling
    try:
        X_batch = torch.stack(images, dim=0)
        Y_batch = torch.stack(labels, dim=0)
    except Exception as e:
        print(f"Stacking error: {e}")
        # Fallback: create tensors manually
        batch_size = len(images)
        img_shape = images[0].shape
        label_shape = labels[0].shape

        X_batch = torch.zeros((batch_size,) + img_shape, dtype=torch.float32)
        Y_batch = torch.zeros((batch_size,) + label_shape, dtype=torch.float32)

        for i, (img, lbl) in enumerate(zip(images, labels)):
            X_batch[i] = img
            Y_batch[i] = lbl

    return X_batch, Y_batch
