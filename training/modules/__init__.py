import logging

from saicinpainting.training.modules.ffc import FFCResNetGenerator
from saicinpainting.training.modules.pix2pixhd import GlobalGenerator, MultiDilatedGlobalGenerator, \
    NLayerDiscriminator, MultidilatedNLayerDiscriminator

from saicinpainting.training.modules.cfgan import Coarse2fine_generator

def make_generator(config, kind, **kwargs):
    logging.info(f'Make generator {kind}')

    if kind == 'pix2pixhd_multidilated':
        return MultiDilatedGlobalGenerator(**kwargs)
    
    if kind == 'pix2pixhd_global':
        return GlobalGenerator(**kwargs)

    if kind == 'ffc_resnet':
        return FFCResNetGenerator(**kwargs)

    if kind=='coarse2fine':
        return Coarse2fine_generator(**kwargs)

    raise ValueError(f'Unknown generator kind {kind}')


def make_discriminator(kind, **kwargs):
    logging.info(f'Make discriminator {kind}')

    if kind == 'pix2pixhd_nlayer_multidilated':
        return MultidilatedNLayerDiscriminator(**kwargs)

    if kind == 'pix2pixhd_nlayer':
        return NLayerDiscriminator(**kwargs)

    raise ValueError(f'Unknown discriminator kind {kind}')
