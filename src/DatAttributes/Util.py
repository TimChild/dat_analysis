import h5py

from src.DatAttributes.DatAttribute import FitInfo


def rows_group_to_all_FitInfos(group: h5py.Group):
    row_group_dict = {}
    for key in group.keys():
        row_id = group[key].attrs.get('row', None)
        if row_id is not None and group[key].attrs.get('description', None) == "Single Parameters of fit":
            row_group_dict[row_id] = group[key]
    fit_infos = [FitInfo()] * len(row_group_dict)
    for key in sorted(row_group_dict.keys()):
        fit_infos[key].init_from_hdf(row_group_dict[key])
    return fit_infos


def fit_group_to_FitInfo(group: h5py.Group):
    assert group.attrs.get('description', None) == "Single Parameters of fit"
    fit_info = FitInfo()
    fit_info.init_from_hdf(group)
    return fit_info

