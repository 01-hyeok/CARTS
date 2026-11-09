def applies(cell,training,heads,batch):
    return cell=='Weather_336' and training=='P100' and heads in (0,1) and batch in (7,28)
