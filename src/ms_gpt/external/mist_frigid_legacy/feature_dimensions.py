"""Feature dimensions used by the FRIGID legacy inference modules."""

from . import chem_utils as utils


class PeakFormula:
    cat_types = {'frags': 0, 'loss': 1, 'ab_loss': 2, 'cls': 3}
    num_inten_bins = 10
    num_types = len(cat_types)
    cls_type = cat_types.get('cls')
    num_adducts = len(utils.ION_LST)

    @classmethod
    def get_num_inten_feats(self, inten_transform):
        """_summary_

        Args:
            inten_transform (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        if inten_transform == 'float':
            inten_feats = 1
        elif inten_transform == 'zero':
            inten_feats = 1
        elif inten_transform == 'log':
            inten_feats = 1
        elif inten_transform == 'cat':
            inten_feats = PeakFormula.num_inten_bins
        else:
            raise NotImplementedError()
        return inten_feats
