import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from sympy import symbols, sympify, diff, sin, cos, log, simplify, factorial
import math
from math import prod

class InterpolationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interpolação")

        # Título
        ttk.Label(root, text="Interpolação de Pontos", font=("Arial", 16)).pack(pady=10)

        # Entrada de pontos
        frame = ttk.Frame(root)
        frame.pack(pady=10)
        ttk.Label(frame, text="Pontos (x,y):").grid(row=0, column=0, padx=5, pady=5)
        self.points_entry = ttk.Entry(frame, width=50)
        self.points_entry.grid(row=0, column=1, padx=5, pady=5)
        self.points_entry.insert(0, "0,0; 1,1; 2,4; 3,9")

        # Entrada de função original
        ttk.Label(frame, text="Função f(x):").grid(row=1, column=0, padx=5, pady=5)
        self.function_entry = ttk.Entry(frame, width=50)
        self.function_entry.grid(row=1, column=1, padx=5, pady=5)
        self.function_entry.insert(0, "sin(x)")

        # Entrada de ponto a ser avaliado
        ttk.Label(frame, text="Ponto x para avaliar:").grid(row=2, column=0, padx=5, pady=5)
        self.x_eval_entry = ttk.Entry(frame, width=20)
        self.x_eval_entry.grid(row=2, column=1, padx=5, pady=5)

        # Escolha do método
        ttk.Label(frame, text="Método:").grid(row=3, column=0, padx=5, pady=5)
        self.method = ttk.Combobox(frame, values=["Linear", "Quadrática", "Lagrange"], state="readonly")
        self.method.grid(row=3, column=1, padx=5, pady=5)
        self.method.current(0)

        # Botões
        button_frame = ttk.Frame(root)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Interpolar", command=self.interpolate).grid(row=0, column=0, padx=10)
        ttk.Button(button_frame, text="Limpar", command=self.clear).grid(row=0, column=1, padx=10)
        ttk.Button(button_frame, text="Calcular Y para X", command=self.calculate_y_for_x).grid(row=1, column=0, padx=10, pady=5)
        ttk.Button(button_frame, text="Diferenças Divididas e Polinômio", command=self.calculate_div_diff_and_polynomial).grid(row=0, column=2, padx=10)
        ttk.Button(button_frame, text="Diferenças Finitas e Polinômio", command=self.calculate_finite_diff_and_polynomial).grid(row=1, column=1, padx=10, pady=5)

        self.result_label = ttk.Label(root, text="", font=("Arial", 12), foreground="blue")
        self.result_label.pack(pady=10)

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

        try:
            x_eval = float(self.x_eval_entry.get())
        except ValueError:
            messagebox.showerror("Erro", "Ponto x para avaliar inválido.")
            return

        # Pontos para interpolação
        x_new = np.linspace(min(x), max(x), 100)

        if method == "Linear":
            coeffs = np.polyfit(x, y, 1)
            f = np.poly1d(coeffs)
        elif method == "Quadrática":
            coeffs = np.polyfit(x, y, 2)
            f = np.poly1d(coeffs)
        elif method == "Lagrange":
            f = lagrange(x, y)
        else:
            messagebox.showerror("Erro", "Método de interpolação desconhecido.")
            return

        y_new = f(x_new)
        y_eval = f(x_eval)

        # Calcular erro de truncamento teórico
        trunc_error = self.calculate_truncation_error(x, x_eval)

        # Exibir função, valor avaliado e erro de truncamento
        self.result_label.config(
            text=f"Função: {f}\nValor em x={x_eval}: y={y_eval:.4f}\nErro Teórico: E≈{trunc_error:.4e}"
        )

        # Plotar o resultado
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, "o", label="Pontos Dados")
        plt.plot(x_new, y_new, "-", label=f"Interpolação ({method})")
        plt.scatter(x_eval, y_eval, color="red", label=f"Ponto Avaliado ({x_eval}, {y_eval:.4f})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Interpolação - Método: {method}")
        plt.legend()
        plt.grid()
        plt.show()

    def newton_polynomial(self, x, y, diff_table):
        """
        Constrói e retorna o polinômio de Newton a partir das diferenças divididas.
        """
        n = len(x)
        terms = [f"{y[0]:.4f}"]  # O primeiro termo é apenas o y_0
        polynomial = f"f(x) = {y[0]:.4f}"

        # Constrói os termos do polinômio de Newton
        for k in range(1, n):
            term = f"({x[k-1]} - x)"
            for i in range(k-1):
                term = f"({x[i]} - x)" + term
            term = f"({diff_table[0, k]:.4f}) * " + term
            terms.append(term)
            polynomial += f" + {diff_table[0, k]:.4f} * " + term

        return polynomial, terms


    def evaluate_and_simplify_polynomial(self, x, y, x_eval, diff_table):
        """
        Avalia e simplifica o polinômio de Newton no ponto x_eval.
        """
        n = len(x)
        # Inicializa o polinômio de Newton
        polynomial = 0
        terms = []
        
        # Construa o polinômio de Newton
        for k in range(n):
            term = diff_table[0, k]
            for i in range(k):
                term *= (x_eval - x[i])
            terms.append(term)
            polynomial += term
        
        # Simplificar o polinômio usando sympy
        x_sym = symbols('x')
        polynomial_expr = 0
        for k in range(n):
            term = diff_table[0, k]
            for i in range(k):
                term *= (x_sym - x[i])
            polynomial_expr += term
        
        # Simplifica o polinômio
        simplified_polynomial = simplify(polynomial_expr)

        # Avaliar o polinômio no ponto x_eval
        y_eval = simplified_polynomial.subs(x_sym, x_eval)

        # Exibir o polinômio simplificado
        return simplified_polynomial, y_eval



    def calculate_div_diff_and_polynomial(self):
        x, y = self.parse_points()
        if x is None or y is None:
            return

        # Calcular as diferenças divididas
        div_diff_text, diff_table = self.div_dif(x, y)

        # Ponto de avaliação
        try:
            x_eval = float(self.x_eval_entry.get())
        except ValueError:
            messagebox.showerror("Erro", "Ponto x para avaliar inválido.")
            return

        # Calcular o polinômio de Newton simplificado e avaliá-lo
        polynomial_expr, y_eval = self.evaluate_and_simplify_polynomial(x, y, x_eval, diff_table)

        # Exibir as diferenças divididas, o polinômio e o valor avaliado
        div_diff_text += f"\n\nPolinômio de Newton: {polynomial_expr}\n\n"
        div_diff_text += f"Valor do polinômio em x={x_eval}: y={y_eval:.4f}"

        self.result_label.config(
            text=f"Diferenças Divididas e Polinômio de Newton:\n{div_diff_text}"
        )

    def finite_differences(self, x, y):
        """
        Calcula as diferenças finitas para a interpolação.
        """
        n = len(x)
        diff_table = np.zeros((n, n))
        diff_table[:, 0] = y  # Primeira coluna são os y's originais

        # Preenchendo a tabela de diferenças finitas
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
        """
        Constrói e retorna o polinômio de diferenças finitas.
        """
        n = len(x)
        terms = [f"{y[0]:.4f}"]  # O primeiro termo é apenas o y_0
        polynomial = f"f(x) = {y[0]:.4f}"

        # Constrói os termos do polinômio de diferenças finitas
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
        """
        n = len(x)
        x_sym = symbols('x')
        
        # Inicializa o polinômio de Gregory-Newton
        polynomial = y[0]
        polynomial_expr = y[0]  # Inicialização do polinômio como o primeiro valor y0
        
        terms = [f"{y[0]:.4f}"]  # O primeiro termo é apenas y0

        # Construa os termos do polinômio de Gregory-Newton
        for k in range(1, n):
            term = diff_table[0, k]
            for i in range(k):
                term *= (x_sym - x[i])
            
            terms.append(f"{diff_table[0, k]:.4f} * " + "*".join([f"(x - {x[i]})" for i in range(k)]))
            polynomial_expr += term  # Adiciona o termo ao polinômio completo
            
            # Para avaliação numérica, somamos o valor de cada termo
            polynomial += diff_table[0, k] * prod([(x_eval - x[i]) for i in range(k)])

        # Simplificar o polinômio usando sympy
        simplified_polynomial = simplify(polynomial_expr)
        
        # Avaliar o polinômio no ponto x_eval
        y_eval = simplified_polynomial.subs(x_sym, x_eval)

        # Exibir o polinômio simplificado
        return simplified_polynomial, y_eval, terms
    def calculate_finite_diff_and_polynomial(self):
        x, y = self.parse_points()
        if x is None or y is None:
            return

        # Calcular as diferenças finitas
        finite_diff_text, diff_table = self.finite_differences(x, y)

        # Ponto de avaliação
        try:
            x_eval = float(self.x_eval_entry.get())
        except ValueError:
            messagebox.showerror("Erro", "Ponto x para avaliar inválido.")
            return

        # Calcular o polinômio de diferenças finitas simplificado e avaliá-lo
        polynomial_expr, y_eval, terms = self.evaluate_and_simplify_polynomial_gregory_newton(x, y, x_eval, diff_table)

        # Exibir as diferenças finitas, o polinômio e o valor avaliado
        finite_diff_text += f"\n\nPolinômio de Diferenças Finitas: {polynomial_expr}\n\n"
        finite_diff_text += f"Valor do polinômio em x={x_eval}: y={y_eval:.4f}"

        self.result_label.config(
            text=f"Diferenças Finitas e Polinômio:\n{finite_diff_text}"
        )



    def calculate_y_for_x(self):
        try:
            # Obter valores de X
            x_vals = self.points_entry.get().strip().split(";")
            x_vals = [float(p.split(",")[0]) for p in x_vals]
            function = self.function_entry.get()
            f_sym = sympify(function, locals={"sin": sin, "cos": cos, "log": log})
            x_sym = symbols("x")

            # Calcular os valores de Y para os pontos X
            y_vals = [f_sym.subs(x_sym, x_val) for x_val in x_vals]

            # Exibir os resultados
            y_vals_str = ", ".join([f"f({x}) = {y.evalf():.4f}" for x, y in zip(x_vals, y_vals)])
            self.result_label.config(text=f"Valores de Y para X: {y_vals_str}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao calcular os valores de Y: {e}")

    def calculate_truncation_error(self, x, x_eval):
        try:
            # Obter função original do usuário
            user_function = self.function_entry.get()
            f_sym = sympify(user_function, locals={"sin": sin, "cos": cos, "log": log})
            x_sym = symbols("x")

            # Calcular derivada de ordem n+1
            n = len(x) - 1
            derivative = diff(f_sym, x_sym, n + 1)

            # Avaliar derivada no ponto médio do intervalo
            xi = (max(x) + min(x)) / 2
            derivative_value = derivative.subs(x_sym, xi)

            # Calcular produto dos termos (x_eval - xi)
            product = np.prod([x_eval - xi for xi in x])

            # Calcular erro teórico
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
        """
        Calcula as diferenças divididas de Newton para interpolação.
        """
        n = len(x)
        diff_table = np.zeros((n, n))
        diff_table[:, 0] = y  # Primeira coluna são os y's originais

        # Preenchendo a tabela de diferenças divididas
        for j in range(1, n):
            for i in range(n - j):
                diff_table[i, j] = (diff_table[i+1, j-1] - diff_table[i, j-1]) / (x[i+j] - x[i])

        # Criando a string para mostrar as diferenças divididas de forma ordenada
        div_diff_text = ""
        for j in range(1, n):
            for i in range(n - j):
                div_diff_text += f"Fórmula para Δ{j}({i}): Δ^{j}f({i}) = {diff_table[i, j]:.4f}\n"
                
        return div_diff_text, diff_table  # Retorna as diferenças divididas em formato legível
    



if __name__ == "__main__":
    root = tk.Tk()
    app = InterpolationApp(root)
    root.mainloop()
