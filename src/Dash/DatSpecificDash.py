"""
All dash things specific to Dat analysis should be implemented here. BaseClasses should be general to any Dash app.

"""
from src.Dash.BaseClasses import BasePageLayout, BaseMain, BaseSideBar
from src.Dash.DatPlotting import OneD, TwoD, ThreeD
from typing import Optional
import abc


# Dash layouts for Dat Specific
class DatDashPageLayout(BasePageLayout, abc.ABC):
    pass


class DatDashMain(BaseMain, abc.ABC):
    pass


class DatDashSideBar(BaseSideBar, abc.ABC):
    pass


# Plotting classes for Dash specific
class DashOneD(OneD):
    def save_to_dat(self, fig, name: Optional[str] = None, sub_group_name: Optional[str] = None, overwrite=True):
        if not sub_group_name:
            sub_group_name = 'Dash'
        super().save_to_dat(fig, name, sub_group_name, overwrite)


class DashTwoD(TwoD):
    def save_to_dat(self, fig, name: Optional[str] = None, sub_group_name: Optional[str] = None, overwrite=True):
        if not sub_group_name:
            sub_group_name = 'Dash'
        super().save_to_dat(fig, name, sub_group_name, overwrite)


class DashThreeD(ThreeD):
    def save_to_dat(self, fig, name: Optional[str] = None, sub_group_name: Optional[str] = None, overwrite=True):
        if not sub_group_name:
            sub_group_name = 'Dash'
        super().save_to_dat(fig, name, sub_group_name, overwrite)
