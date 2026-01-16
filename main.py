from manim import *
import numpy as np

class GroupFinalPresentation(Scene):
    def construct(self):
        self.camera.background_color = "#0f172a"


        group_title = Text("Group 7", font_size=48, color=TEAL)
        group_title.to_edge(UP, buff=1.0)

        names = VGroup(
            Text("Uchasara Huarachi, Diego David", font_size=32),
            Text("Roque Castillo, Franco Nicolas", font_size=32),
            Text("Montalvo Anaya, Diego Andres", font_size=32)
        ).arrange(DOWN, buff=0.5)
        names.next_to(group_title, DOWN, buff=1)

        video_title = Text("Linear vs Polynomial Regression", font_size=40, color=YELLOW)
        video_title.next_to(names, DOWN, buff=1.5)

        self.play(Write(group_title), run_time=0.8)
        self.play(FadeIn(names, lag_ratio=0.3), run_time=1.5)
        self.play(Write(video_title), run_time=1)
        self.wait(2)

        self.play(
            FadeOut(group_title), 
            FadeOut(names), 
            FadeOut(video_title),
            run_time=1
        )

        
        theory_title_1 = Text("1. Linear Regression Theory", font_size=36, color=BLUE)
        theory_title_1.to_corner(UL)
        self.play(Write(theory_title_1))

        hypothesis_text = Text("Hypothesis:", font_size=24, color=GREY_B).shift(UP*2 + LEFT*2)
        eq_hyp = MathTex(r"h_\theta(x) = w_0 + w_1 x", font_size=36)
        eq_hyp.next_to(hypothesis_text, DOWN)

        loss_text = Text("Loss Function (MSE):", font_size=24, color=GREY_B).next_to(eq_hyp, DOWN, buff=0.5).align_to(hypothesis_text, LEFT)
        eq_loss = MathTex(
            r"J(w) = \frac{1}{2n} \sum_{i=1}^{n} (h_\theta(x^{(i)}) - y^{(i)})^2",
            font_size=36, color=RED_B
        )
        eq_loss.next_to(loss_text, DOWN)

        grad_text = Text("Gradient Update (Derivative):", font_size=24, color=GREY_B).next_to(eq_loss, DOWN, buff=0.5).align_to(loss_text, LEFT)
        
        eq_grad = MathTex(
            r"\frac{\partial J}{\partial w} = \frac{1}{n} \sum (h_\theta(x) - y) \cdot x",
            font_size=36, color=YELLOW
        )
        eq_grad.next_to(grad_text, DOWN)
        
        eq_update = MathTex(
            r"w_{new} = w_{old} - \alpha \frac{\partial J}{\partial w}",
            font_size=36
        ).next_to(eq_grad, DOWN)

        self.play(FadeIn(hypothesis_text), Write(eq_hyp))
        self.wait(1)
        self.play(FadeIn(loss_text), Write(eq_loss))
        self.wait(1)
        self.play(FadeIn(grad_text), Write(eq_grad))
        self.play(Write(eq_update))
        self.wait(3)

        self.play(
            FadeOut(theory_title_1), FadeOut(hypothesis_text), FadeOut(eq_hyp),
            FadeOut(loss_text), FadeOut(eq_loss), 
            FadeOut(grad_text), FadeOut(eq_grad), FadeOut(eq_update)
        )


        axes = Axes(
            x_range=[-3, 3, 1], y_range=[-2, 2, 1],
            x_length=9, y_length=5.5,
            axis_config={"color": GREY_A, "include_numbers": True, "font_size": 20}
        ).shift(DOWN * 0.5)

        np.random.seed(42)
        x = np.linspace(-2.5, 2.5, 30)
        y_true = np.arctan(x) + np.random.normal(0, 0.15, len(x))

        dots = VGroup(*[Dot(axes.c2p(vx, vy), radius=0.06, color=WHITE) for vx, vy in zip(x, y_true)])
        
        panel = VGroup(
            Text("Iterations:", font_size=24, color=GREY_B),
            Integer(0, font_size=24, color=YELLOW),
            Text("MSE Loss:", font_size=24, color=RED_B),
            DecimalNumber(0, num_decimal_places=4, font_size=24, color=RED)
        ).arrange(DOWN, aligned_edge=LEFT).to_corner(UR)
        
        iter_val = panel[1]
        mse_val = panel[3]

        self.play(Create(axes), FadeIn(dots), FadeIn(panel))

        label_lin = MathTex(r"\text{Model: } w_0 + w_1 x", color=BLUE, font_size=32).to_corner(UL).shift(RIGHT)
        self.play(Write(label_lin))

        w0, w1 = -1.0, -0.5
        alpha = 0.05
        line = axes.plot(lambda t: w0 + w1 * t, x_range=[-3, 3], color=BLUE)
        self.play(Create(line))

        for i in range(40):
            y_pred = w0 + w1 * x
            loss = np.mean((y_true - y_pred)**2)
            error = y_pred - y_true
            w0 -= alpha * np.mean(error)
            w1 -= alpha * np.mean(error * x)
            
            new_line = axes.plot(lambda t: w0 + w1 * t, x_range=[-3, 3], color=BLUE)
            self.play(
                Transform(line, new_line),
                ChangeDecimalToValue(mse_val, loss),
                ChangeDecimalToValue(iter_val, i+1),
                run_time=0.08, rate_func=linear
            )

        result_text_1 = Text("Result: Underfitting (High Bias)", color=RED, font_size=28).next_to(label_lin, DOWN)
        self.play(FadeIn(result_text_1))
        self.wait(2)

        self.play(FadeOut(axes), FadeOut(dots), FadeOut(panel), FadeOut(line), FadeOut(label_lin), FadeOut(result_text_1))

        
        theory_title_2 = Text("2. Polynomial Regression Theory", font_size=36, color=GREEN)
        theory_title_2.to_corner(UL)
        self.play(Write(theory_title_2))

        feature_text = Text("Feature Expansion:", font_size=24, color=GREY_B).shift(UP*2 + LEFT*2)
        expansion = MathTex(
            r"x \rightarrow [1, x, x^2, x^3, \dots, x^d]", 
            font_size=36
        ).next_to(feature_text, DOWN)

        hyp_poly_text = Text("Polynomial Hypothesis:", font_size=24, color=GREY_B).next_to(expansion, DOWN, buff=0.5).align_to(feature_text, LEFT)
        eq_poly = MathTex(
            r"h_\theta(x) = \sum_{j=0}^{d} w_j x^j", 
            font_size=38, color=GREEN
        ).next_to(hyp_poly_text, DOWN)

        grad_poly_text = Text("Gradient for each weight w_j:", font_size=24, color=GREY_B).next_to(eq_poly, DOWN, buff=0.5).align_to(hyp_poly_text, LEFT)
        
        eq_grad_poly = MathTex(
            r"\frac{\partial J}{\partial w_j} = \frac{1}{n} \sum (h_\theta(x) - y) \cdot x^j",
            font_size=36, color=YELLOW
        ).next_to(grad_poly_text, DOWN)

        self.play(FadeIn(feature_text), Write(expansion))
        self.wait(1)
        self.play(FadeIn(hyp_poly_text), Write(eq_poly))
        self.wait(1)
        self.play(FadeIn(grad_poly_text), Write(eq_grad_poly))
        self.wait(3)

        self.play(
            FadeOut(theory_title_2), FadeOut(feature_text), FadeOut(expansion),
            FadeOut(hyp_poly_text), FadeOut(eq_poly), 
            FadeOut(grad_poly_text), FadeOut(eq_grad_poly)
        )
        
        self.play(Create(axes), FadeIn(dots), FadeIn(panel))
        self.play(iter_val.animate.set_value(0), mse_val.animate.set_value(0))

        label_poly3 = MathTex(r"\text{Case A: Degree } d=3", color=GREEN, font_size=32).to_corner(UL).shift(RIGHT)
        self.play(Write(label_poly3))

        x_mean, x_std = np.mean(x), np.std(x)
        x_n = (x - x_mean) / x_std
        
        degree = 3
        X_mat = np.vander(x_n, degree + 1, increasing=True)
        W = np.array([0.0, -0.5, 0.0, 0.1])
        lr = 0.08

        def poly_func(w_curr):
            return lambda t: sum(w_curr[j] * ((t - x_mean)/x_std)**j for j in range(len(w_curr)))

        curve3 = axes.plot(poly_func(W), x_range=[-3, 3], color=GREEN)
        self.play(Create(curve3))

        for i in range(50):
            y_pred = X_mat @ W
            loss = np.mean((y_true - y_pred)**2)
            grad = X_mat.T @ (y_pred - y_true) / len(x)
            W -= lr * grad
            
            new_curve3 = axes.plot(poly_func(W), x_range=[-3, 3], color=GREEN)
            self.play(
                Transform(curve3, new_curve3),
                ChangeDecimalToValue(mse_val, loss),
                ChangeDecimalToValue(iter_val, i+1),
                run_time=0.08, rate_func=linear
            )

        result_text_2 = Text("Result: Good Fit (Balanced)", color=GREEN, font_size=28).next_to(label_poly3, DOWN)
        self.play(FadeIn(result_text_2))
        self.wait(2)

        self.play(FadeOut(curve3), FadeOut(label_poly3), FadeOut(result_text_2))
        
        
        self.play(iter_val.animate.set_value(0), mse_val.animate.set_value(0))

        label_poly5 = MathTex(r"\text{Case B: Degree } d=5", color=ORANGE, font_size=32).to_corner(UL).shift(RIGHT)
        self.play(Write(label_poly5))

        degree = 5
        X_mat5 = np.vander(x_n, degree + 1, increasing=True)
        W5 = np.zeros(degree + 1)
        lr = 0.05

        curve5 = axes.plot(poly_func(W5), x_range=[-3, 3], color=ORANGE)
        self.play(Create(curve5))

        for i in range(70):
            y_pred = X_mat5 @ W5
            loss = np.mean((y_true - y_pred)**2)
            grad = X_mat5.T @ (y_pred - y_true) / len(x)
            W5 -= lr * grad
            
            new_curve5 = axes.plot(poly_func(W5), x_range=[-3, 3], color=ORANGE)
            self.play(
                Transform(curve5, new_curve5),
                ChangeDecimalToValue(mse_val, loss),
                ChangeDecimalToValue(iter_val, i+1),
                run_time=0.08, rate_func=linear
            )

        result_text_3 = Text("Result: Low Error (High Capacity)", color=ORANGE, font_size=28).next_to(label_poly5, DOWN)
        self.play(FadeIn(result_text_3))
        
        rect = SurroundingRectangle(panel, color=YELLOW, buff=0.2)
        self.play(Create(rect))
        
        self.wait(4)

        final_text = Text("Thanks for watching!", font_size=36)
        self.play(FadeOut(axes), FadeOut(dots), FadeOut(panel), FadeOut(curve5), FadeOut(label_poly5), FadeOut(result_text_3), FadeOut(rect))
        self.play(Write(final_text))
        self.wait(2)