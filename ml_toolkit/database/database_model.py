import logging
import typing

import sqlalchemy as sa
from sqlalchemy import (Column, DateTime, Float, Integer, String)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
logger = logging.getLogger(__name__)


class BaseModel(Base):
    __abstract__ = True
    def __repr__(self) -> str:
        return self._repr(id=self.id)

    def _repr(self, **fields: typing.Dict[str, typing.Any]) -> str:
        '''
        Helper for __repr__
        '''
        field_strings = []
        at_least_one_attached_attribute = False
        for key, field in fields.items():
            try:
                field_strings.append(f'{key}={field!r}')
            except sa.orm.exc.DetachedInstanceError:
                field_strings.append(f'{key}=DetachedInstanceError')
            else:
                at_least_one_attached_attribute = True
        if at_least_one_attached_attribute:
            return f"<{self.__class__.__name__}({','.join(field_strings)})>"
        return f"<{self.__class__.__name__} {id(self)}>"

    @classmethod
    def type(cls):
        return cls.__name__

    @classmethod
    def declarativeClass(cls):
        return cls

class Results(BaseModel):
    __tablename__ = 'results'
    id = Column(Integer, primary_key=True)
    grid_id = Column(Integer)
    exp = Column(Integer)
    ##
    model = Column(String)
    m_p_k = Column(String)
    m_p_v = Column(String)
    ##
    dataset = Column(String)
    d_p_k = Column(String)
    d_p_v = Column(String)
    ##
    splits = Column(String)
    nrepeat = Column(Integer)
    nfold = Column(Integer)
    epochs = Column(Integer)
    ##
    fold = Column(Integer)
    repeat = Column(Integer)
    ##
    train_accuracy = Column(Float)
    val_accuracy = Column(Float)
    test_accuracy = Column(Float)
    #
    train_accmacro = Column(Float)
    val_accmacro = Column(Float)
    test_accmacro = Column(Float)
    #
    train_f1macro = Column(Float)
    val_f1macro = Column(Float)
    test_f1macro = Column(Float)
    #
    train_epoch = Column(Integer)
    val_epoch = Column(Integer)
    test_epoch = Column(Integer)
    #
    train_confmx = Column(String)
    val_confmx = Column(String)
    test_confmx = Column(String)
    #
    train_timestamp = Column(DateTime)
    val_timestamp = Column(DateTime)
    test_timestamp = Column(DateTime)

    def __repr__(self):
        return self._repr(id=self.id, grid_id=self.grid_id, exp=self.exp, model=self.model,m_p_k=self.m_p_k,m_p_v=self.m_p_v,
            dataset=self.dataset,d_p_k=self.d_p_k,d_p_v=self.d_p_v,splits=self.splits,
            nrepeat=self.nrepeat, nfold=self.nfold, epochs=self.epochs,fold=self.fold,repeat=self.repeat,
            train_accuracy=self.train_accuracy,val_accuracy=self.val_accuracy,test_accuracy=self.test_accuracy,
            train_accmacro=self.train_accmacro,val_accmacro=self.val_accmacro,test_accmacro=self.test_accmacro,
            train_f1macro=self.train_f1macro,val_f1macro=self.val_f1macro,test_f1macro=self.test_f1macro,
            train_epoch=self.train_epoch,val_epoch=self.val_epoch,test_epoch=self.test_epoch,
            train_confmx=self.train_confmx,val_confmx=self.val_confmx,test_confmx=self.test_confmx,
            train_timestamp=self.train_timestamp,val_timestamp=self.val_timestamp,test_timestamp=self.test_timestamp,
            )
