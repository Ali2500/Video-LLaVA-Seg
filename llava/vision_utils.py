
def get_resize_padding_params(img_h, img_w, tgt_size, pad_mode):
    assert pad_mode in ('center', 'topleft')
    pad_left = pad_right = pad_top = pad_bottom = 0

    if img_h > img_w:
        img_h = tgt_size
        img_w = int(round((img_w / img_h) * tgt_size))
        if pad_mode == 'center':
            pad_left = (img_h - img_w) // 2
            pad_right = img_h - img_w - pad_left
        else:
            pad_right = img_h - img_w
    else:
        img_w = tgt_size
        img_h = int(round((img_h / img_w) * tgt_size))
        if pad_mode == 'center':
            pad_top = (img_w - img_h) // 2
            pad_bottom = img_w - img_h - pad_top
        else:
            pad_bottom = img_w - img_h

    return (img_h, img_w), (pad_left, pad_right, pad_top, pad_bottom)
