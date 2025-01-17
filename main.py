import tkinter as tk
import pandas as pd
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from sympy import symbols, sympify, diff, sin, cos, log, simplify, factorial
import math
import time
import psutil

class InterpolationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interpolação")
        frame = ttk.Frame(root)
        frame.pack(pady=10)
        ttk.Label(frame, text="Pontos (x,y):").grid(row=0, column=0, padx=5, pady=5)
        self.points_entry = ttk.Entry(frame, width=50)
        self.points_entry.grid(row=0, column=1, padx=5, pady=5)
        self.points_entry.insert(0, "0,0; 1,1; 2,4; 3,9")
        ttk.Label(frame, text="Função f(x):").grid(row=1, column=0, padx=5, pady=5)
        self.function_entry = ttk.Entry(frame, width=50)
        self.function_entry.grid(row=1, column=1, padx=5, pady=5)
        self.function_entry.insert(0, "sin(x)")
        ttk.Label(frame, text="Ponto x para avaliar:").grid(row=2, column=0, padx=5, pady=5)
        self.x_eval_entry = ttk.Entry(frame, width=20)
        self.x_eval_entry.grid(row=2, column=1, padx=5, pady=5)
        ttk.Label(frame, text="Método:").grid(row=3, column=0, padx=5, pady=5)
        self.method = ttk.Combobox(frame, values=["Linear", "Quadrática", "Lagrange"], state="readonly")
        self.method.grid(row=3, column=1, padx=5, pady=5)
        self.method.current(0)
        button_frame = ttk.Frame(root)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Interpolar", command=self.interpolate).grid(row=0, column=0, padx=10)
        ttk.Button(button_frame, text="Limpar", command=self.clear).grid(row=0, column=1, padx=10)
        ttk.Button(button_frame, text="Calcular Y para X", command=self.calculate_y_for_x).grid(row=1, column=0, padx=10, pady=5)
        ttk.Button(button_frame, text="Diferenças Divididas e Polinômio", command=self.calculate_div_diff_and_polynomial).grid(row=0, column=2, padx=10)
        ttk.Button(button_frame, text="Diferenças Finitas e Polinômio", command=self.calculate_finite_diff_and_polynomial).grid(row=1, column=1, padx=10, pady=5)
        ttk.Button(button_frame, text="Carregar CSV", command=self.load_csv).grid(row=1, column=2, padx=10, pady=5)
        self.result_label = ttk.Label(root, text="", font=("Arial", 12), foreground="blue")
        self.result_label.pack(pady=10)

    def load_csv(self):
        """Carrega pontos de um arquivo CSV e atualiza o campo de entrada."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                data = pd.read_csv(file_path, header=None)  # Lê o arquivo CSV sem cabeçalho
                points = "; ".join([f"{row[0]},{row[1]}" for row in data.values])  # Formata os pontos
                self.points_entry.delete(0, 'end')
                self.points_entry.insert(0, points)
            except Exception as e:
                self.result_label.config(text=f"Erro ao carregar CSV: {e}")

    def parse_points(self):
        try:
            points = self.points_entry.get().strip().split(";")
            points = [tuple(map(float, p.split(","))) for p in points]
            x, y = zip(*points)
            return np.array(x), np.array(y)
        except ValueError:
            messagebox.showerror("Erro", "Entrada de pontos inválida. Use o formato x,y; x,y...")
            return None, None
    def interpolate(self):
        x, y = self.parse_points()
        if x is None or y is None:
            return

        method = self.method.get()
        x_eval = None
        try:
            x_eval = float(self.x_eval_entry.get())
        except ValueError:
            pass
        x_new = np.linspace(min(x), max(x), 100)
        f = None
        trunc_error = None
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Usar memória em MB
        if method == "Linear":
            coeffs = np.polyfit(x, y, 1)
            f = np.poly1d(coeffs)
            trunc_error = self.truncation_error_linear(x, y)
        elif method == "Quadrática":
            coeffs = np.polyfit(x, y, 2)
            f = np.poly1d(coeffs)
            trunc_error = self.truncation_error_quadratic(x, y)
        elif method == "Lagrange":
            f = lagrange(x, y)
            trunc_error = self.truncation_error_lagrange(x, y)
        else:
            messagebox.showerror("Erro", "Método de interpolação desconhecido.")
            return
        y_new = f(x_new)
        y_eval = None
        if x_eval is not None:
            y_eval = f(x_eval)
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Memória após execução em MB
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        polynomial_text = self.format_polynomial(f, method)
        result_text = f"Função: {polynomial_text}\n"
        if y_eval is not None:
            result_text += f"Valor em x={x_eval}: y={y_eval:.4f}\n"
        if trunc_error is not None:
            result_text += f"Erro Teórico: E≈{trunc_error:.4e}\n"
        result_text += f"Tempo de Execução: {execution_time:.4f} segundos\n"
        result_text += f"Memória Usada: {memory_used:.4f} MB\n"
        self.result_label.config(text=result_text)
        output_file = "resultados_saida.txt"
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(result_text)
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, "o", label="Pontos Dados")
        plt.plot(x_new, y_new, "-", label=f"Interpolação ({method})")
        if x_eval is not None:
            plt.scatter(x_eval, y_eval, color="red", label=f"Ponto Avaliado ({x_eval}, {y_eval:.4f})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Interpolação - Método: {method}")
        plt.legend()
        plt.grid()
        plt.show()

    def format_polynomial(self, f, method):
        """
        Formata a função interpoladora no formato "ax^2 + bx + c".
        Para funções de Lagrange, utiliza uma representação simplificada.
        """
        if isinstance(f, np.poly1d):
            terms = []
            coeffs = f.coefficients
            degree = len(coeffs) - 1
            for i, coef in enumerate(coeffs):
                if abs(coef) > 1e-10:  # Ignorar coeficientes próximos de zero
                    power = degree - i
                    if power == 0:
                        terms.append(f"{coef:.4g}")
                    elif power == 1:
                        terms.append(f"{coef:.4g}x")
                    else:
                        terms.append(f"{coef:.4g}x^{power}")
            return " + ".join(terms).replace("+ -", "- ")
        elif method == "Lagrange":
            return "Interpolação de Lagrange (pode ser complexa para exibição direta)"
        else:
            return "Função não formatável"


    def truncation_error_linear(self, x, y):
        """
        Calcula o erro de truncamento para interpolação linear.
        """
        h = max(x) - min(x)
        return h**2 / 8 

    def truncation_error_quadratic(self, x, y):
        """
        Calcula o erro de truncamento para interpolação quadrática.
        """
        h = max(x) - min(x)
        max_third_derivative = 0

        for i in range(1, len(x) - 2):  # Garante que i+2 e i-1 estejam dentro dos limites
            third_derivative = (y[i + 2] - 3 * y[i + 1] + 3 * y[i] - y[i - 1]) / ((x[i + 1] - x[i]) ** 3)
            max_third_derivative = max(max_third_derivative, abs(third_derivative))

        return max_third_derivative * h**3 / 24


    def truncation_error_lagrange(self, x, y):
        """
        Calcula o erro de truncamento para interpolação de Lagrange.
        """
        h = max(x) - min(x)
        n = len(x)
        max_derivative = 0  
        for i in range(n - 1):
            derivative = abs((y[i + 1] - y[i]) / (x[i + 1] - x[i]))
            max_derivative = max(max_derivative, derivative)
        return max_derivative * h**(n + 1) / (n + 1)  
    def newton_polynomial(self, x, y, diff_table, xi):
        n = len(x)
        result = y[0]
        for k in range(1, n):
            term = diff_table[0, k]  
            for i in range(k):
                term *= (xi - x[i]) 
            result += term  
        return result
    def evaluate_and_simplify_polynomial(self, x, y, x_eval, diff_table):
        n = len(x)
        polynomial = 0
        terms = []
        for k in range(n):
            term = diff_table[0, k]
            for i in range(k):
                term *= (x_eval - x[i])
            terms.append(term)
            polynomial += term
        x_sym = symbols('x')
        polynomial_expr = 0
        for k in range(n):
            term = diff_table[0, k]
            for i in range(k):
                term *= (x_sym - x[i])
            polynomial_expr += term
        simplified_polynomial = simplify(polynomial_expr)
        y_eval = simplified_polynomial.subs(x_sym, x_eval)
        return simplified_polynomial, y_eval

    def calculate_div_diff_and_polynomial(self):
        x, y = self.parse_points()
        if x is None or y is None:
            return

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        div_diff_text, diff_table = self.div_dif(x, y)
        
        try:
            x_eval = float(self.x_eval_entry.get())
        except ValueError:
            messagebox.showerror("Erro", "Ponto x para avaliar inválido.")
            return

        polynomial_expr, y_eval = self.evaluate_and_simplify_polynomial(x, y, x_eval, diff_table)
        
        div_diff_text += f"\n\nPolinômio de Newton: {polynomial_expr}\n\n"
        div_diff_text += f"Valor do polinômio em x={x_eval}: y={y_eval:.4f}"

        self.result_label.config(
            text=f"Diferenças Divididas e Polinômio de Newton:\n{div_diff_text}"
        )

        x_new = np.linspace(min(x), max(x), 100)

        y_new = [self.newton_polynomial(x, y, diff_table, xi) for xi in x_new]

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        execution_time = end_time - start_time
        memory_used = end_memory - start_memory

        result_text = div_diff_text
        result_text += f"\n\nTempo de Execução: {execution_time:.4f} segundos\n"
        result_text += f"Memória Usada: {memory_used:.4f} MB\n"

        self.result_label.config(
            text=result_text
        )
        output_file = "resultados_saida.txt"
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(result_text)

        plt.figure(figsize=(8, 5))
        plt.plot(x, y, 'o', label="Pontos Dados")
        plt.plot(x_new, y_new, '-', label="Interpolação de Newton")
        if x_eval is not None:
            plt.scatter(x_eval, y_eval, color="red", label=f"Ponto Avaliado ({x_eval}, {y_eval:.4f})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Interpolação - Polinômio de Newton")
        plt.legend()
        plt.grid()
        plt.show()

    def finite_differences(self, x, y):
        n = len(x)
        diff_table = np.zeros((n, n))
        diff_table[:, 0] = y 
        for j in range(1, n):
            for i in range(n - j):
                diff_table[i, j] = diff_table[i+1, j-1] - diff_table[i, j-1]

        # Criando a string para mostrar as diferenças finitas de forma ordenada
        finite_diff_text = ""
        for j in range(1, n):
            for i in range(n - j):
                finite_diff_text += f"Fórmula para Δ{j}({i}): Δ^{j}f({i}) = {diff_table[i, j]:.4f}\n"
        return finite_diff_text, diff_table

    def finite_difference_polynomial(self, x, y, diff_table):
        n = len(x)
        terms = [f"{y[0]:.4f}"] 
        polynomial = f"f(x) = {y[0]:.4f}"
        for k in range(1, n):
            term = f"({x[k-1]} - x)"
            for i in range(k-1):
                term = f"({x[i]} - x)" + term
            term = f"({diff_table[0, k]:.4f}) * " + term
            terms.append(term)
            polynomial += f" + {diff_table[0, k]:.4f} * " + term

        return polynomial, terms

    def evaluate_and_simplify_polynomial_gregory_newton(self, x, y, x_eval, diff_table):
        """
        Avalia e simplifica o polinômio de Gregory-Newton no ponto x_eval.
        Também calcula o erro de truncamento.
        """
        n = len(x)
        x_sym, z = symbols('x z')
        h = x[1] - x[0] 
        z_expr = (x_sym - x[0]) / h 
        polynomial_expr = y[0]
        terms = [f"{y[0]:.4f}"]
        for k in range(1, n):
            delta_y = diff_table[0, k]
            term = delta_y / factorial(k)
            for i in range(k):
                term *= (z - i)
            polynomial_expr += term
            terms.append(f"({delta_y:.4f} / {factorial(k)}) * " + "*".join([f"(z - {i})" for i in range(k)]))
        simplified_polynomial = simplify(polynomial_expr)
        polynomial_in_x = simplified_polynomial.subs(z, (x_sym - x[0]) / h)
        simplified_polynomial_in_x = simplify(polynomial_in_x)
        z_eval = (x_eval - x[0]) / h  
        y_eval = simplified_polynomial.subs(z, z_eval)
        truncation_error = h**(n + 1) / factorial(n + 1)
        for i in range(n + 1):
            truncation_error *= (z_eval - i)
        f_derivative_approx = diff_table[0, n-1]
        truncation_error *= f_derivative_approx
        return simplified_polynomial_in_x, y_eval, terms, truncation_error



    def calculate_finite_diff_and_polynomial(self):
        x, y = self.parse_points()
        if x is None or y is None:
            return

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        finite_diff_text, diff_table = self.finite_differences(x, y)

        try:
            x_eval = float(self.x_eval_entry.get())
        except ValueError:
            messagebox.showerror("Erro", "Ponto x para avaliar inválido.")
            return

        polynomial_expr, y_eval, terms, truncation_error = self.evaluate_and_simplify_polynomial_gregory_newton(x, y, x_eval, diff_table)

        finite_diff_text += f"\n\nPolinômio de Diferenças Finitas: {polynomial_expr}\n\n"
        finite_diff_text += f"Valor do polinômio em x={x_eval}: y={y_eval:.4f}"
        finite_diff_text += f"\nValor do Erro de Truncamento:{truncation_error}"

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        execution_time = end_time - start_time
        memory_used = end_memory - start_memory

        result_text = finite_diff_text
        result_text += f"\n\nTempo de Execução: {execution_time:.4f} segundos\n"
        result_text += f"Memória Usada: {memory_used:.4f} MB\n"

        self.result_label.config(
            text=result_text
        )
        output_file = "resultados_saida.txt"
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(result_text)

    def calculate_y_for_x(self):
        try:
            x_vals = self.points_entry.get().strip().split(";")
            x_vals = [float(p.split(",")[0]) for p in x_vals]
            function = self.function_entry.get()
            f_sym = sympify(function, locals={"sin": sin, "cos": cos, "log": log})
            x_sym = symbols("x")
            y_vals = [f_sym.subs(x_sym, x_val) for x_val in x_vals]
            y_vals_str = ", ".join([f"f({x}) = {y.evalf():.4f}" for x, y in zip(x_vals, y_vals)])
            self.result_label.config(text=f"Valores de Y para X: {y_vals_str}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao calcular os valores de Y: {e}")

    def calculate_truncation_error(self, x, x_eval):
        try:
            user_function = self.function_entry.get()
            f_sym = sympify(user_function, locals={"sin": sin, "cos": cos, "log": log})
            x_sym = symbols("x")
            n = len(x) - 1
            derivative = diff(f_sym, x_sym, n + 1)
            xi = (max(x) + min(x)) / 2
            derivative_value = derivative.subs(x_sym, xi)
            product = np.prod([x_eval - xi for xi in x])
            trunc_error = abs(derivative_value * product / math.factorial(n + 1))
            return trunc_error
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao calcular o erro de truncamento: {e}")
            return 0
    def clear(self):
        self.points_entry.delete(0, tk.END)
        self.points_entry.insert(0, "")
        self.function_entry.delete(0, tk.END)
        self.function_entry.insert(0, "")
        self.x_eval_entry.delete(0, tk.END)
        self.result_label.config(text="")
    def div_dif(self, x, y):
        n = len(x)
        diff_table = np.zeros((n, n))
        diff_table[:, 0] = y
        for j in range(1, n):
            for i in range(n - j):
                diff_table[i, j] = (diff_table[i+1, j-1] - diff_table[i, j-1]) / (x[i+j] - x[i])
        div_diff_text = ""
        for j in range(1, n):
            for i in range(n - j):
                div_diff_text += f"Fórmula para Δ{j}({i}): Δ^{j}f({i}) = {diff_table[i, j]:.4f}\n"
        return div_diff_text, diff_table  
if __name__ == "__main__":
    root = tk.Tk()
    app = InterpolationApp(root)
    root.mainloop()
