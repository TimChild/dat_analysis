from dash_dashboard.base_classes import BasePageLayout, BaseMain, BaseSideBar
import abc


class DatDashPageLayout(BasePageLayout, abc.ABC):
    top_bar_title = "Tim's Dat Viewer"


class DatDashMain(BaseMain, abc.ABC):
    pass


class DatDashSidebar(BaseSideBar, abc.ABC):
    pass
