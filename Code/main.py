from control import train_model, eval_model
from data import load_TREC_data
from model import GANet
import argparse


def main():
	fix_length = 10
	batch_size = 32
	embedding_length = 100

	TEXT, vocab_size, word_embeddings, train_iter, test_iter = load_TREC_data(batch_size, embedding_length, fix_length)

	learning_rate = 2e-5
	output_size = 2
	hidden_size = 256
	embedding_length = 100
	num_classes = 6
	mlp_out_size = 32
	weights = word_embeddings
	aux_hidden_size = 100
	batch_hidden_size = 100

	model = GANet(batch_size, num_classes, mlp_out_size, vocab_size, embedding_length, weights, biDirectional_aux=True, biDirectional_backbone=True)

	for epoch in range(15):
	    train_loss, train_acc = train_model(model, train_iter, epoch, batch_size)
	    # val_loss, val_acc = eval_model(model, valid_iter)
	    
	    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%')
	    # print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
	    
	test_loss, test_acc = eval_model(model, test_iter, batch_size)
	print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

if __name__ == "__main__":
	main()
