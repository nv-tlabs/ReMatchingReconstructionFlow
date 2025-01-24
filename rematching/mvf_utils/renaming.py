def full_name(short_name):
    if short_name == "P1":
        return "rematching.rematching_loss.DiscreteReMatchingLoss_P1"
    elif short_name == "P3":
        return "rematching.rematching_loss.DiscreteReMatchingLoss_AdaptivePriors_P3"
    elif short_name == "P4":
        return "rematching.rematching_loss.DiscreteReMatchingLoss_AdaptivePriors_P4"
    elif short_name == "P1_P3":
        return "rematching.rematching_loss.DiscreteReMatchingLoss_AdaptivePriors_P1_P3"
    elif short_name == "P1_P4":
        return "rematching.rematching_loss.DiscreteReMatchingLoss_AdaptivePriors_P1_P4"
    elif short_name == "P3_Image":
        return "rematching.rematching_loss.FunctionReMatchingLoss_Image_P3"
    else:
        print("Invalid prior name: "+short_name)