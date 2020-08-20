# Gated Attention

Implementation of the paper : [Not all attention is needed - Gated Attention Network for Sequence Data (GA-Net)](https://arxiv.org/abs/1912.00349) 

##### Flow Diagram for the network:  
There are two networks in the model:  
1. Backbone Network  
2. Auxiliary Network  

![](images/flow_dia.png?raw=true)  

##### Comparison with soft attention network:
Soft Attention gives some attention (low or high) to all the input tokens whereas gated attention network chooses the most important tokens to attend.

![](images/attention.png?raw=true)

##### Gate Probability and gated attention:
Visualization of probability for gate to be open for input token and the actual gated attention weight.  

![](images/gate_prob_and_attn.png?raw=true)  
                                                                                                   
