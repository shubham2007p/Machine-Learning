class Linear_regression():
    def __int__(self , slope , intercept):
        self.slope = slope
        self.intercept =  intercept

    def predict(self ,x_value):
        prediction = (self.slope * x_value) + self.intercept
        return prediction

def calculate_mse(x_values , y_values , model):
    global mega_set
    mega_set= list()
    mse = 0
    for x_value , y_value in zip(x_values, y_values):
        mega_set.append([x_value, y_value])
        prediction = model.predict(x_value)
        error = prediction - y_value
        mse += abs(error)**2

    mse = mse / len(x_values)
    return mse

def optimize_intercept(x_values , y_values ,model , best_model):
    iter_count =1
    while True:
        iter_count +=1
        model.intercept +=1
        current_mse = calculate_mse(x_values , y_values ,model)
        if current_mse < best_model["model"][0]:
            best_model["model"] = [current_mse , model.slope , model.intercept]
            print(best_model)
            continue
        else:
            model.intercept -=1
            if iter_count == 1:
                model.intercept -=1
                current_mse = calculate_mse(x_values , y_values ,model)
                if current_mse < best_model["model"][0]:
                    best_model["model"] = [current_mse , model.slope , model.intercept]
                    print(best_model)
                    continue
                else:
                    model.intercept +=1
                    break
            break

    return

    

def optimize_slope(x_values , y_values , model , best_model):
    iter_count =1
    while True:
        iter_count +=1
        model.slope +=0.1
        current_mse = calculate_mse(x_values , y_values ,model)
        if current_mse < best_model["model"][0]:
            best_model["model"] = [current_mse , model.slope , model.intercept]
            print(best_model)
            continue
        else:
            model.slope -=0.1
            if iter_count == 1:
                model.slope -=0.1
                current_mse = calculate_mse(x_values , y_values ,model)
                if current_mse < best_model["model"][0]:
                    best_model["model"] = [current_mse , model.slope , model.intercept]
                    print(best_model)
                    continue
                else:
                    model.slope +=0.1
                    break
            break

    return
    
def slope_decider(mega_set , model):
    slopes = list()
    prev_pair = [0,0]
    for pair in mega_set:
        slope = (pair[1] - prev_pair[1]) / (pair[0] - prev_pair[0])   
        print(slope) 
        prev_pair = pair
        slopes.append(slope)

    slopes.sort()

    if len(slopes) % 2 == 0:
        index_1 = len(slopes) // 2
        index_2 = index_1 -1
        median_slope = (slopes[index_1] + slopes[index_2]) //2
    else:
        median_slope = slopes[len(slopes)//2]

    model.slope = median_slope
    print(model.slope)
    return


def train_model(x_values , y_values , model , best_model):
    if not best_model:
        current_mse = calculate_mse(x_values, y_values , model)
        best_model["model"] = [current_mse , model.slope , model.intercept]
    
    slope_decider(mega_set , model)
    count = 0
    while True:

        optimize_intercept(x_values , y_values ,model , best_model)
        optimize_slope(x_values , y_values ,model , best_model)

        break
   

    return best_model




def main():
    best_model = dict()

    hours = [1,2,3,4]
    marks = [42,51,61,69]

    x_value = int(input("Enter the hours : "))

    model = Linear_regression()
    model.slope =0 
    model.intercept =0


    best_model = train_model(hours , marks , model , best_model)

    model.slope = best_model["model"][1]
    model.intercept = best_model["model"][2]

    prediction = model.predict(x_value)
    print(best_model)
    print(f"prediction is : {prediction}")




if __name__ == "__main__":
    main()


"""
rewriting code : for now closet got mse is 5. somthing i wanna tell chatgpt but ahh no more scrabbling in chat its not about code or anything now its about the straight lines engineering

1. so to find slope i am running slope decider at first then small stesp in slop and big step in slope and big step inintercept
"""