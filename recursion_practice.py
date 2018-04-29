"""
The purpose of this exercise was to practice recursive functions.
The challenge here is to take a given text,a string of letters, and return all the legitimate sentences, where a
legitimate sentence is one that comprised of words from a given bank. Also it needs to use all the letters in the
text.

There are two main functions:
- splits_text_to_words_recursion: my solution with recursion
- splits_text_to_words_ordinary: my solution without recursion

There are three other support functions:
- find_first_word
- extend_word
- remove_prefix
"""

BANK_OF_WORDS = ['the', 'dog', 'ate', 'cat', 'do', 'gate']
SENTENCE = "thedogatethecat"


def splits_text_to_words_recursion(sentence, bank_of_words):
    """
    The function splits a string of letters into separated words, considering all combinations, based on a bank of
    words. The function is recursive.
    :param sentence: a string of letters without spaces
    :param bank_of_words: a list of legitimate words
    :return: a list of lists, where each list is a sentence separated to words
    """

    first_word = find_first_word(sentence, bank_of_words) # The first word in the sentence
    # The following condition will return None if no word was found
    if first_word == None:
        return "No legitimate sentences were found"
    # The following condition will return [first word] if this is the last word in the sentence
    elif len(first_word) == len(sentence):
        return [first_word]
    else:
        developing_sentences = [[first_word]] # This is the beginning of each developing sencece (greedy model)
        next_word = extend_word(sentence, first_word, bank_of_words) # This is a word the includes the first word
        # (non-greedy)
        # The following condition is for the cases where the first_word is not a part of another next_word)
        if next_word != None:
            developing_sentences.append([next_word])
        updated_developing_sentences = [] # This list will contain the split sentence or sentences (if more than one)
        # This for loop allows the combination of sentences
        for word in [first_word, next_word]:
            # This if condition is for the case that there is no next_word
            if word == None:
                 continue
            remaining_text = remove_prefix(sentence, word) # The text without the word
            # The following line uses the self function to get the remaining text split to words
            remain_text_split_to_words = splits_text_to_words_recursion(remaining_text, bank_of_words)
            # The following for loop takes each word from remain_text_split_to_words and adds to the developing
            # sentences
            for element in remain_text_split_to_words:
                # following condition is for the last word in the sentence that returns as a string
                if type(element) == str:
                    element = [element]
                new_sentence = developing_sentences[developing_sentences.index([word])] + element
                updated_developing_sentences.append(new_sentence)

    return updated_developing_sentences



# The following version is the non-recursion
def splits_text_to_words_ordinary(sentence, bank_of_words):
    """
    The function splits a string of letters into separated words, considering all combinations, based on a bank of
    words. The function is a non-recursive.
    :param sentence: a string of letters without spaces
    :param bank_of_words: a list of legitimate words
    :return: a list of lists, where each list is a sentence separated to words
    """

    sentence_queue = [] # This is a queue where all developing sentences will be stored
    legitimate_sentences = [] # Once a full legitimate sentence is found, it will be stores here
    # The following condition allows the situation where no legitimate sentence was found (no first word)
    if find_first_word(sentence, bank_of_words) != None:
        sentence_queue.append([find_first_word(sentence, bank_of_words)]) # First word must be in the queue before the
                                                                          # while loop
        # This while loop runs as long as there is something in the queue
        while len(sentence_queue) > 0:
            sentence_in_work = sentence_queue[0] # The function takes always the first item in the queue to work on
            sentence_so_far = ''.join(sentence_in_work) # Combines the words to one string, to allow having the
                                                        # remaining text of the sentence
            remaining_text = remove_prefix(sentence, sentence_so_far)
            # This conditon allows the distinguishing between the last word and any other word
            if len(remaining_text) > 0:
                next_word = find_first_word(remaining_text, bank_of_words)
                # The condition takes the next word (greedy), adds it to the sentence_in_work and then to the queue
                if next_word != None:
                    sentence_in_work.append(find_first_word(remaining_text, bank_of_words))
                    sentence_queue.append(sentence_in_work)
                # The next three lines finds a loger word that includes next_word (non-greedy)
                last_word_in_sentence_so_far = sentence_in_work[-1]
                last_word_with_remaining = last_word_in_sentence_so_far + remaining_text
                next_word_instead_last_word = extend_word(last_word_with_remaining, last_word_in_sentence_so_far, bank_of_words)
                # The condition takes the word (non-greedy), adds it to the sentence_in_work and then to the queue
                if next_word_instead_last_word != None:
                    sentence_in_work_without_last_word = sentence_in_work[:-1]
                    sentence_in_work_without_last_word.append(next_word_instead_last_word)
                    sentence_queue.append(sentence_in_work_without_last_word)
            else:
                legitimate_sentences.append((sentence_in_work))
            sentence_queue.pop(0) # must pop sentence_in_work to allow end the while loop
        return legitimate_sentences
    else:
        return "No legitimate sentences were found"


# The following 3 functions are supportive functions

def find_first_word(sentence, bank_of_words):
    """
    The function finds the first word in a sentence, given a bank of words. It adds a letter each time to the
    developing word, and when it matches a word in the bank it returns it
    :param sentence: a string of letters
    :param bank_of_words: a list of legitimate words
    :return: a string with the first word
    """

    letter_gathering = []
    for letter in sentence:
        potential_word = ''.join(letter_gathering) + letter
        if potential_word in bank_of_words:
            return potential_word
        else:
            letter_gathering.append(letter)


def extend_word(sentence, short_word, bank_of_words):
    """
    Given a word, the functions finds a longer word that incorporates the first one.
    :param sentence: a string of letters, begins with the short_word
    :param short_word: a string with the short word
    :param bank_of_words:
    :return: a string with the longer word
    """
    remaining = sentence.lstrip(short_word)
    for letter in remaining:
        potential_word = short_word + letter
        if potential_word in bank_of_words:
            return potential_word

def remove_prefix(text, pre):
    """
    Takes a text, and removes a shorter string from the beginning in case the text begins with the shorter one
    :param text: a string
    :param pre: a string
    :return:
    """
    if text.startswith(pre):
        return text[len(pre):]


print("recursive")
print("---------")
print(splits_text_to_words_recursion(SENTENCE, BANK_OF_WORDS))
print()
print("non-recursive")
print("-------------")
print(splits_text_to_words_ordinary(SENTENCE, BANK_OF_WORDS))

