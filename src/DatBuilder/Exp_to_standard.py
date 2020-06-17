from src.Configs import Main_Config as cfg
from src.CoreUtil import verbose_message
import logging

logger = logging.getLogger(__name__)


def temp_from_json(jsondict, fridge='ls370'):
    if 'BF Small' in jsondict.keys():
        try:
            temps = temp_from_bfsmall(jsondict['BF Small'])
        except KeyError as e:
            print(jsondict)
            raise e
        return temps
    else:

        logger.info(f'Verbose[][temp_from_json] - Did not find "BF Small" in json')

    return None


def srs_from_json(jsondict, id, srs_type='srs830'):
    if 'SRS_' + str(id) in jsondict.keys():
        srsdict = jsondict['SRS_' + str(id)]
        srsdata = {'gpib': srsdict['gpib_address'],
                   'out': srsdict['amplitude V'],
                   'tc': srsdict['time_const ms'],
                   'freq': srsdict['frequency Hz'],
                   'phase': srsdict['phase deg'],
                   'sens': srsdict['sensitivity V'],
                   'harm': srsdict['harmonic'],
                   'CH1readout': srsdict.get('CH1readout', None)
                   }
    else:
        srsdata = None
    return srsdata


def mag_from_json(jsondict, id, mag_type='ls625'):
    if 'LS625 Magnet Supply' in jsondict.keys():  # FIXME: This probably only works if there is 1 magnet ONLY!
        mag_dict = jsondict['LS625 Magnet Supply']  #  FIXME: Might just be able to pop entry out then look again
        magname = mag_dict.get('variable name', None)  # Will get 'magy', 'magx' etc
        if magname[-1:] == id:  # compare last letter
            mag_data = {'field': mag_dict['field mT'],
                        'rate': mag_dict['rate mT/min']
                        }
        else:
            mag_data = None
    else:
        mag_data = None
    return mag_data


def temp_from_bfsmall(tempdict):
    tempdata = {'mc': tempdict.get('MC K', None),
                'still': tempdict.get('Still K', None),
                'fourk': tempdict.get('4K Plate K', None),
                'mag': tempdict.get('Magnet K', None),
                'fiftyk': tempdict.get('50K Plate K', None)}
    return tempdata