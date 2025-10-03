import mynetwork
import loader

def main():
    
    net = mynetwork.MyNetwork()
    net.sgd()

    # TODO: plot model performance on 10 test examples

if __name__ == '__main__':
    main()