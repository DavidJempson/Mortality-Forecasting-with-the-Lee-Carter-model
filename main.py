import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

#Get data from files into python arrays:
#(could have created repeatedly called function, but this will be more efficient computationally)
malesDeathData = []
femalesDeathData = []
currentYear = 1950
currentMaleCol = []
currentFemaleCol = []
with open("Deaths_1x1.txt") as deathsFile:
    next(deathsFile)
    next(deathsFile)
    next(deathsFile)
    for line in deathsFile:
        splittedLine = [x for x in line[:-1].split(" ") if x]
        if  int(splittedLine[0]) == currentYear:
            currentMaleCol.append(float(splittedLine[3]))
            currentFemaleCol.append(float(splittedLine[2]))
        else:
            malesDeathData.append(currentMaleCol[:111])
            femalesDeathData.append(currentFemaleCol[:111])
            currentMaleCol = [float(splittedLine[3])]
            currentFemaleCol = [float(splittedLine[2])]
            currentYear += 1
    currentMaleCol.append(float(splittedLine[3]))
    currentFemaleCol.append(float(splittedLine[2]))
    malesDeathData.append(currentMaleCol[:111])
    femalesDeathData.append(currentFemaleCol[:111])

malesExposuresData = []
femalesExposuresData = []
currentYear = 1950
currentMaleCol = []
currentFemaleCol = []
with open("Exposures_1x1.txt") as exposuresFile:
    next(exposuresFile)
    next(exposuresFile)
    next(exposuresFile)
    for line in exposuresFile:
        splittedLine = [x for x in line[:-1].split(" ") if x]
        if  int(splittedLine[0]) == currentYear:
            currentMaleCol.append(float(splittedLine[3]))
            currentFemaleCol.append(float(splittedLine[2]))
        else:
            malesExposuresData.append(currentMaleCol[:111])
            femalesExposuresData.append(currentFemaleCol[:111])
            currentMaleCol = [float(splittedLine[3])]
            currentFemaleCol = [float(splittedLine[2])]
            currentYear += 1
    currentMaleCol.append(float(splittedLine[3]))
    currentFemaleCol.append(float(splittedLine[2]))
    malesExposuresData.append(currentMaleCol[:111])
    femalesExposuresData.append(currentFemaleCol[:111])

#Move to numpy arrays for easier manipulation:
male_deaths_array = np.array(malesDeathData).transpose()
male_exposures_array = np.array(malesExposuresData).transpose()
female_deaths_array = np.array(femalesDeathData).transpose()
female_exposures_array = np.array(femalesExposuresData).transpose()

#Manipulate the data into log mortality rates tables:

#Numpy will give error even when things work perfectly, so include these lines:
with np.errstate(divide='ignore', invalid='ignore'):
    male_mortality_rates = np.where(male_exposures_array > 0, male_deaths_array / male_exposures_array, np.nan)
    female_mortality_rates = np.where(female_exposures_array > 0, female_deaths_array / female_exposures_array, np.nan)

male_mortality_rates[male_mortality_rates == 0] = np.nan
female_mortality_rates[female_mortality_rates == 0] = np.nan

log_male_mortality_rates = np.log(male_mortality_rates)
log_female_mortality_rates = np.log(female_mortality_rates)

#Estimates parameters for Lee-Carter model from the inputted log mortality rates array
def estimate_parameters(log_mortality_rates):
    '''
    Args:
        log_mortality_rates (np.array): array of logs of central mortality rates
    Returns:
        parameters for Lee-Carter model
        a_x (np.array), b_x (np.array), k_t (np.array)
    '''
    #Estimate a_X
    a_x = np.nanmean(log_mortality_rates, axis=1)

    #Create centred matrix for SVD:
    centered_matrix = log_mortality_rates - a_x[:, np.newaxis]
    #Replace NaNs with 0s for SVD:
    centered_matrix[np.isnan(centered_matrix)] = 0

    #Decompose matrix with SVD ---
    U, S, Vt = np.linalg.svd(centered_matrix, full_matrices=False)

    #Get paramters we want:
    b_x = U[:, 0]
    k_t = S[0] * Vt[0, :]

    #Apply constraints:
    #need sum(bx) = 1
    sum_bx = np.sum(b_x)
    b_x = b_x / sum_bx
    k_t = k_t * sum_bx

    #need sum(kt) = 0
    mean_kt = np.mean(k_t)
    k_t = k_t - mean_kt
    a_x = a_x + b_x * mean_kt

    return a_x, b_x, k_t

male_a_x, male_b_x, male_k_t = estimate_parameters(log_male_mortality_rates)
female_a_x, female_b_x, female_k_t = estimate_parameters(log_female_mortality_rates)

#Uses ARIMA model to forecast change in k_t in the future
def forecast(k_t):
    '''
    Args:
        k_t (np.array): parameter for Lee-Carter mode
    Returns:
        forecasted_k_t (np.array): predicted values of k_t in future
    '''
    #How many years into future to forecast:
    forecast_steps = 110

    #Use statsmodels to model data with ARIMA(0, 1, 0)
    model = sm.tsa.arima.ARIMA(k_t, order=(0, 1, 0), trend='t')
    model_fit = model.fit()

    #Can print summary for diagnostics:
    #print(model_fit.summary())

    #Extract the data we want:
    forecast_result = model_fit.get_forecast(steps=forecast_steps)
    forecasted_k_t = forecast_result.predicted_mean
    #conf_int = forecast_result.conf_int(alpha=0.05) # 95% confidence interval
    return forecasted_k_t

male_forecast_k_t = forecast(male_k_t)
female_forecast_k_t = forecast(female_k_t)

#Creates a generation (AKA cohort) life for specific birth year:
def create_generation_life_table(birth_year, a_x, b_x, full_k_t, start_year=1950, max_age=109):
    '''
    Args:
        birth_year (int): The birth year of the generation to be followed.
        a_x (np.array): parameter from Lee-Carter model
        b_x (np.array): parameter from Lee-Carter model
        full_k_t (np.array): The complete kt time series (historical + forecast).
        start_year (int): The first year of the kt series.
        max_age (int): The maximum age for the table.

    Returns:
        pd.DataFrame: A pandas DataFrame representing the generation life table, with following columns:
        m_x: central mortality rate
        q_x: mortality rate
    '''
    ages = np.arange(0, max_age + 1)
    num_ages = len(ages)

    #Get the generation's mortality rates from the data:
    generation_m_x = np.zeros(num_ages)
    for age in ages:
        #Find index in table for age and birth year, to access data we need:
        forecast_year = birth_year + age
        year_index = forecast_year - start_year

        #If year is in our range, reconstrct central mortality rate:
        if 0 <= year_index < len(full_k_t):
            k_t_for_age = full_k_t[year_index]
            log_mx = a_x[age] + b_x[age] * k_t_for_age
            generation_m_x[age] = np.exp(log_mx)
        else:
            #If we don't have data for the age/year, fill with NaNs
            generation_m_x[age] = np.nan

    #Create the life table:
    q_x = generation_m_x / (1 + 0.5 * generation_m_x)
    q_x[-1] = 1.0

    #Put in pandas dataframe:
    life_table_df = pd.DataFrame({
        'Age': ages,
        'Year': ages + birth_year,
        'm_x': generation_m_x,
        'q_x': q_x,
    })
    return life_table_df

#Calculates the price of a premium with given inputs
def calculatePremium(sex, age, sumAssured, term, interestRate = 4):
    '''
    Args:
        sex (str): can be 'm' or 'f' for male or female
        age (int): age of person buying premium
        sumAssured (float): sum payed out on death
        term (int): number of years the policy lasts
        interestRate (float): assumed interest rate over term
    Returns:
        yearly price of policy (float)
    '''
    #Create life table for person with given sex, age, etc:
    if sex == 'm':
        life_table = create_generation_life_table(2025-age, male_a_x, male_b_x, np.concatenate([male_k_t, male_forecast_k_t]))
    elif sex== 'f':
        life_table = create_generation_life_table(2025-age, female_a_x, female_b_x, np.concatenate([female_k_t, female_forecast_k_t]))
    else:
        print("ERROR with inputted sex")
        return -1
    
    #Get mortality rates for remaining years of life
    q_x_list = life_table['q_x'].to_list()[age:]

    #Used in policy price calculation:
    p_x_list = [1-q_x_list[0]]
    for q_x in q_x_list:
        p_x_list.append(p_x_list[-1]*(1-q_x))

    #Find EPV of benefit (A) and premium (A)
    A = 0
    a = 0
    for k in range(term):
        A += p_x_list[k]*q_x_list[k]/((1+(interestRate/100))**(k+1))
        a += p_x_list[k]/((1+(interestRate/100))**k)
    
    #Return yearly price:
    return sumAssured*A/a

#EXAMPLE USAGE:
premiumCost = calculatePremium(sex = 'm', age = 35, sumAssured = 250000, term=20)
print(f"Monthly cost of policy: £{premiumCost/12:.2f}")