from dataclasses import dataclass

class TransactionManager():
    def __init__(self,qmax) -> None:
        self.qmax = qmax
        self.events: list[TransactionEvent] = list()
        
    def add(self, action: float, close: float) -> float:
        position = self.get_position()
        d_position = action - position
        d_shares = round(self.qmax*d_position) #? Does it? -> this causes an exceed of the position value at times.
        d_position_round = d_shares/self.qmax 

        if d_shares !=0:
            event = TransactionEvent(position_change=d_position_round, price_per_share=close, transaction_price=d_shares*close)
            self.events.append(event)

            reward = -(d_shares*close)
        else:
            reward = 0


        return reward
    
    def get_position(self):
        position = 0
        for event in self.events:
            position += event.position_change
        
        return position

    def get_gain_at_position(self, position, close) -> float:
        total_purchase_price = 0
        s_position = 0

        for event in self.events:
            if isinstance(event, TransactionEvent):
                s1_position = s_position + event.position_change

                if s1_position*s_position<0:
                    total_purchase_price = 0 + s1_position*self.qmax*event.price_per_share
                else:
                    total_purchase_price += event.transaction_price
                
                
                s_position = s1_position
        
        d_position = s_position - position
        liquidation_value = d_position*self.qmax*close
        gain = liquidation_value - total_purchase_price

        return gain


@dataclass
class TransactionEvent:
    position_change: int
    price_per_share: float
    transaction_price: float
