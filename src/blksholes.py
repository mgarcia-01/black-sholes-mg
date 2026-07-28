import math
import csv
from dataclasses import dataclass
from typing import List

@dataclass
class OptionTestData:
    """A data container representing a single test case row."""
    is_call: bool
    S: float
    K: float
    T: float
    r: float
    sigma: float
    expected_price: float


def standard_normal_pdf(x: float) -> float:
    """
    Calculates the probability density function (PDF) for the standard normal distribution.
    
    Args:
        x (float): The value to evaluate.
        
    Returns:
        float: The probability density at x.
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def standard_normal_cdf(x: float) -> float:
    """
    Calculates the cumulative distribution function (CDF) for the standard normal distribution.
    This uses the Abramowitz and Stegun approximation, maintaining an absolute error of less than 7.5e-8.
    
    Args:
        x (float): The upper bound of the integral.
        
    Returns:
        float: The probability that a standard normal random variable is less than or equal to x.
    """
    # Constants for the approximation
    p = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    # Symmetry implies CDF(-x) = 1 - CDF(x)
    if x < 0.0:
        return 1.0 - standard_normal_cdf(-x)

    t = 1.0 / (1.0 + p * x)
    polynomial = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
    
    return 1.0 - standard_normal_pdf(x) * polynomial


def calculate_option_price(is_call: bool, S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculates the price of a European option using the Black-Scholes formula.
    
    Algorithm:
    1. Calculate d1, which represents the probability-weighted expected payout of the asset.
    2. Calculate d2, which represents the probability that the option will expire in the money.
    3. Apply the Black-Scholes formula for either a Call or a Put based on the boolean flag.
    
    Args:
        is_call (bool): True to calculate a Call option price, False for a Put option.
        S (float): The current spot price of the underlying asset.
        K (float): The strike price of the option.
        T (float): The time to expiration in years (e.g., 6 months = 0.5).
        r (float): The annualized risk-free interest rate (e.g., 5% = 0.05).
        sigma (float): The annualized volatility of the underlying asset's returns.
        
    Returns:
        float: The theoretical price of the option.
    """
    # Handle edge case where time to expiration is zero
    if T <= 0.0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)

    # Calculate d1 and d2
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if is_call:
        # Call Option Pricing Formula: C = S * N(d1) - K * e^(-rT) * N(d2)
        return S * standard_normal_cdf(d1) - K * math.exp(-r * T) * standard_normal_cdf(d2)
    else:
        # Put Option Pricing Formula: P = K * e^(-rT) * N(-d2) - S * N(-d1)
        return K * math.exp(-r * T) * standard_normal_cdf(-d2) - S * standard_normal_cdf(-d1)


def read_data_from_csv(file_path: str) -> List[OptionTestData]:
    """
    Reads option data from a CSV file.
    Expects columns: Type(CALL/PUT), S, K, T, r, sigma, ExpectedPrice
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        List[OptionTestData]: A list of OptionTestData objects parsed from the file.
    """
    test_cases = []
    
    try:
        # with statement ensures the file is safely closed after execution
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            
            # Skip the header row
            next(reader, None)
            
            for row in reader:
                if not row or not "".join(row).strip():
                    continue  # Skip empty rows
                
                is_call = row[0].strip().upper() == "CALL"
                S = float(row[1].strip())
                K = float(row[2].strip())
                T = float(row[3].strip())
                r = float(row[4].strip())
                sigma = float(row[5].strip())
                expected_price = float(row[6].strip())
                
                test_cases.append(OptionTestData(is_call, S, K, T, r, sigma, expected_price))
                
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except ValueError as e:
        print(f"Error parsing numeric data in the file: {e}")
        
    return test_cases


def run_test_cases(test_cases: List[OptionTestData], tolerance: float):
    """
    Runs the calculation model against a list of test cases and reports passes/failures.
    
    Args:
        test_cases (List[OptionTestData]): The list of test data to run.
        tolerance (float): The acceptable margin of error (epsilon) for floating-point comparisons.
    """
    passed = 0
    total = len(test_cases)

    print(f"Running {total} Test Cases...\n")

    for i, data in enumerate(test_cases, start=1):
        calculated_price = calculate_option_price(
            data.is_call, data.S, data.K, data.T, data.r, data.sigma
        )
        
        # Check if the difference between calculated and expected is within our tolerance
        diff = abs(calculated_price - data.expected_price)
        is_pass = diff <= tolerance
        
        if is_pass:
            passed += 1
            print(f"[PASS] Test {i} | Expected: {data.expected_price:.4f} | Calculated: {calculated_price:.4f}")
        else:
            print(f"[FAIL] Test {i} | Expected: {data.expected_price:.4f} | Calculated: {calculated_price:.4f} | Diff: {diff:.4f}")

    print(f"\nResults: {passed}/{total} Passed")


if __name__ == "__main__":
    # Option 1: Create mock test data directly in memory
    memory_tests = [
        OptionTestData(is_call=True, S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, expected_price=10.4506),
        OptionTestData(is_call=False, S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, expected_price=5.5741)
    ]
    
    print("--- Running In-Memory Tests ---")
    run_test_cases(memory_tests, tolerance=0.001)

    print("\n--- Running CSV File Tests ---")
    # Option 2: Load from a file (Uncomment and replace "options_data.csv" with your actual file path)
    #file_path = "data/options_data.csv"
    file_path = "data/syn_options.csv"
    file_tests = read_data_from_csv(file_path)
    if file_tests:
        run_test_cases(file_tests, tolerance=0.001)