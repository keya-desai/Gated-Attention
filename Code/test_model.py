# ''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
# test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
# test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."

# test_sen1 = TEXT.preprocess(test_sen1)
# test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]

# test_sen2 = TEXT.preprocess(test_sen2)
# test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]

# test_sen = np.asarray(test_sen1)
# test_sen = torch.LongTensor(test_sen)
# test_tensor = Variable(test_sen, volatile=True)
# test_tensor = test_tensor.cuda()
# model.eval()
# output = model(test_tensor, 1)
# out = F.softmax(output, 1)
# if (torch.argmax(out[0]) == 1):
#     print ("Sentiment: Positive")
# else:
#     print ("Sentiment: Negative")

# Tester function

test_sen1 = "Who killed Gandhi?"   # Class: HUM
 
def test_sentence(test_sen):

  test_sen = TEXT.preprocess(test_sen)
  print(test_sen)
  test_sen = [[TEXT.vocab.stoi[x] for x in test_sen]]
  print(text_sen)

  test_sen = np.asarray(test_sen)
  test_sen = torch.LongTensor(test_sen)
  test_tensor = Variable(test_sen, volatile=True)

  print(test_tensor)
  model.eval()
  prediction, g_t = model(test_tensor, is_train = False)
  print(prediction)
  out_class = torch.argmax(prediction)
  return out_class

x = test_sentence(test_sen1)
print(x)