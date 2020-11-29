"""
All dash things specific to Dat analysis should be implemented here. BaseClasses should be general to any Dash app.

"""
from src.Dash.BaseClasses import BasePageLayout, BaseMain, BaseSideBar
import abc


class DatDashPageLayout(BasePageLayout, abc.ABC):
    pass


class DatDashMain(BaseMain, abc.ABC):
    pass


class DatDashSideBar(BaseSideBar, abc.ABC):
    pass


