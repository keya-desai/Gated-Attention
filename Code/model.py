import torch
from torch import nn
from submodels import AuxiliaryNet, BackboneNet, MLP


class GANet(torch.nn.Module):
		def __init__(self, batch_size, num_classes, mlp_out_size, vocab_size, embedding_length, weights, aux_hidden_size = 100, backbone_hidden_size = 100, biDirectional_aux = False, biDirectional_backbone = False):
			super(GANet, self).__init__() 
			"""
			Arguments
			---------
			batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
			output_size : 6 = (For TREC dataset)
			hidden_sie : Size of the hidden_state of the LSTM   (// Later BiLSTM)
			vocab_size : Size of the vocabulary containing unique words
			embedding_length : Embeddding dimension of GloVe word embeddings
			weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

			--------

			"""

			self.batch_size = batch_size
			self.num_classes = num_classes
			self.vocab_size = vocab_size
			self.embedding_length = embedding_length
			self.aux_hidden_size = aux_hidden_size
			self.backbone_hidden_size = backbone_hidden_size 
			self.mlp_out_size = mlp_out_size
			self.biDirectional_aux = biDirectional_aux
			self.biDirectional_backbone = biDirectional_backbone

			self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
			self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)

			self.auxiliary = AuxiliaryNet(self.batch_size, self.aux_hidden_size, self.embedding_length, self.biDirectional_aux)
			self.backbone = BackboneNet(self.batch_size, self.backbone_hidden_size, self.embedding_length, self.biDirectional_backbone)
			if(self.biDirectional_backbone):
				self.mlp = MLP(self.backbone_hidden_size * 2, self.mlp_out_size)
				self.FF = nn.Linear(self.backbone_hidden_size * 2,num_classes)
			else:
				self.mlp = MLP(self.backbone_hidden_size, self.mlp_out_size)
				self.FF = nn.Linear(self.backbone_hidden_size,num_classes)
			self.softmax = nn.Softmax()
			

		def masked_Softmax(self, logits, mask):
			# print("type of mask", type(mask))
			# print("gt size", mask.shape)
			mask_bool = mask >0
			logits[mask_bool] = float('-inf')
			return torch.softmax(logits, dim=1)	

		def forward(self,input_sequence, is_train = True):
			input_ = self.word_embeddings(input_sequence)
			g_t = self.auxiliary(input_, is_train)
			out_lstm = self.backbone(input_)

			if is_train:
				e_t = self.mlp(out_lstm)
				alpha = self.softmax(e_t)
			else:
				e_t = self.mlp(out_lstm)               # change if possible!
				alpha = self.masked_Softmax(e_t, g_t)

			c_t = torch.bmm(alpha.transpose(1,2), out_lstm)
			logits = self.FF(c_t)
			final_output = self.softmax(logits)
			# final_output = final_output.max(2)[1]
			final_output = final_output.squeeze(1)

			return final_output, g_t
