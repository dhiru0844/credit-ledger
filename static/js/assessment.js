(() => {

    const form =
        document.getElementById("assessmentForm");

    const submitButton =
        document.getElementById("assessmentSubmit");

    const submitText =
        document.querySelector(".submit-text");

    const errorBox =
        document.getElementById("assessmentError");

    const resultSection =
        document.getElementById("assessmentResult");

    const incomeInput =
        document.getElementById("person_income");

    const amountInput =
        document.getElementById("loan_amnt");

    const percentInput =
        document.getElementById("loan_percent_income");

    const resultProbability =
        document.getElementById("resultProbability");

    const resultThreshold =
        document.getElementById("resultThreshold");

    const resultVerdict =
        document.getElementById("resultVerdict");

    const resultBadge =
        document.getElementById("resultBadge");

    const newAssessmentButton =
        document.getElementById("newAssessmentButton");


    /* =====================================
       AUTO CALCULATE LOAN / INCOME RATIO
    ====================================== */

    function calculateLoanRatio() {

        const income =
            parseFloat(incomeInput.value);

        const loan =
            parseFloat(amountInput.value);


        if (
            Number.isFinite(income) &&
            income > 0 &&
            Number.isFinite(loan) &&
            loan >= 0
        ) {

            percentInput.value =
                (loan / income).toFixed(2);

        }

    }


    incomeInput.addEventListener(
        "input",
        calculateLoanRatio
    );


    amountInput.addEventListener(
        "input",
        calculateLoanRatio
    );



    /* =====================================
       ERROR
    ====================================== */

    function showError(message) {

        errorBox.textContent = message;

        errorBox.classList.add("show");

    }


    function clearError() {

        errorBox.textContent = "";

        errorBox.classList.remove("show");

    }



    /* =====================================
       LOADING
    ====================================== */

    function setLoading(loading) {

        submitButton.disabled = loading;


        if (loading) {

            submitText.textContent =
                "Assessing your profile…";

        } else {

            submitText.textContent =
                "Assess my risk";

        }

    }



    /* =====================================
       GET FORM DATA
    ====================================== */

    function getPayload() {

        return {

            person_age:
                parseInt(
                    document.getElementById(
                        "person_age"
                    ).value,
                    10
                ),

            person_income:
                parseFloat(
                    incomeInput.value
                ),

            person_home_ownership:
                document.getElementById(
                    "person_home_ownership"
                ).value,

            person_emp_length:
                parseFloat(
                    document.getElementById(
                        "person_emp_length"
                    ).value
                ),

            loan_intent:
                document.getElementById(
                    "loan_intent"
                ).value,

            loan_grade:
                document.getElementById(
                    "loan_grade"
                ).value,

            loan_amnt:
                parseFloat(
                    amountInput.value
                ),

            loan_int_rate:
                parseFloat(
                    document.getElementById(
                        "loan_int_rate"
                    ).value
                ),

            loan_percent_income:
                parseFloat(
                    percentInput.value
                ),

            cb_person_default_on_file:
                document.getElementById(
                    "cb_person_default_on_file"
                ).value,

            cb_person_cred_hist_length:
                parseInt(
                    document.getElementById(
                        "cb_person_cred_hist_length"
                    ).value,
                    10
                )

        };

    }



    /* =====================================
       SHOW RESULT
    ====================================== */

    function showResult(data) {

        const probability =
            Number(data.default_probability);

        const threshold =
            Number(data.threshold);

        const isHighRisk =
            Number(data.default_prediction) === 1;


        resultProbability.textContent =
            `${(probability * 100).toFixed(1)}%`;


        resultThreshold.textContent =
            `${(threshold * 100).toFixed(1)}%`;


        resultVerdict.textContent =
            data.Result;


        resultBadge.textContent =
            isHighRisk
                ? "HIGH RISK"
                : "LOW RISK";


        resultBadge.classList.remove(
            "high",
            "low"
        );


        resultBadge.classList.add(
            isHighRisk
                ? "high"
                : "low"
        );


        const notesBox = document.getElementById("resultNotes");
        notesBox.innerHTML = "";
        const notes = Array.isArray(data.notes) ? data.notes : [];
        notes.forEach((text) => {
            const li = document.createElement("li");
            li.textContent = text;
            notesBox.appendChild(li);
        });
        notesBox.hidden = notes.length === 0;

        resultSection.hidden = false;


        resultSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    }



    /* =====================================
       SUBMIT
    ====================================== */

    form.addEventListener(
        "submit",
        async (event) => {

            event.preventDefault();

            clearError();

            calculateLoanRatio();


            if (!form.checkValidity()) {

                form.reportValidity();

                return;

            }


            setLoading(true);


            try {

                const payload =
                    getPayload();


                const response =
                    await fetch(
                        "/predict",
                        {
                            method: "POST",

                            headers: {
                                "Content-Type":
                                    "application/json"
                            },

                            body:
                                JSON.stringify(
                                    payload
                                )
                        }
                    );


                const data =
                    await response.json();


                if (!response.ok) {

                    throw new Error(
                        data.detail ||
                        "Unable to assess this profile."
                    );

                }


                showResult(data);

            }

            catch (error) {

                console.error(error);

                showError(
                    error.message ||
                    "Something went wrong. Please make sure the API is running."
                );

            }

            finally {

                setLoading(false);

            }

        }
    );



    /* =====================================
       NEW ASSESSMENT
    ====================================== */

    newAssessmentButton.addEventListener(
        "click",
        () => {

            form.reset();

            resultSection.hidden = true;

            clearError();

            window.scrollTo({
                top: 0,
                behavior: "smooth"
            });

        }
    );

})();