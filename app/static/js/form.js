// Credit Risk AI - Multi-Step Form Handler

function loanApplicationForm() {
    return {
        step: 1,
        totalSteps: 4,
        loading: false,
        
        // Form data
        formData: {
            full_name: '',
            age: '',
            employment_type: 'salaried',
            employment_length: '',
            home_ownership: 'rent',
            annual_income: '',
            existing_debt: '0',
            num_credit_lines: '1',
            credit_history_length: '',
            late_payments: '0',
            bankruptcies: '0',
            loan_amount: '',
            loan_purpose: 'personal',
            loan_tenure: '36'
        },
        
        // Validation errors
        errors: {},
        
        // Go to next step
        nextStep() {
            if (this.validateStep()) {
                if (this.step < this.totalSteps) {
                    this.step++;
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                }
            }
        },
        
        // Go to previous step
        prevStep() {
            if (this.step > 1) {
                this.step--;
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        },
        
        // Go to specific step
        goToStep(s) {
            if (s <= this.step || this.validateStep()) {
                this.step = s;
            }
        },
        
        // Validate current step
        validateStep() {
            this.errors = {};
            
            if (this.step === 1) {
                if (!this.formData.full_name.trim()) this.errors.full_name = 'Full name is required';
                if (!this.formData.age || this.formData.age < 18 || this.formData.age > 100) this.errors.age = 'Age must be between 18 and 100';
                if (!this.formData.employment_length && this.formData.employment_length !== 0) this.errors.employment_length = 'Employment length is required';
            }
            
            if (this.step === 2) {
                if (!this.formData.annual_income || this.formData.annual_income <= 0) this.errors.annual_income = 'Valid income is required';
                if (this.formData.existing_debt === '' || this.formData.existing_debt < 0) this.errors.existing_debt = 'Existing debt cannot be negative';
                if (!this.formData.credit_history_length && this.formData.credit_history_length !== 0) this.errors.credit_history_length = 'Credit history length is required';
            }
            
            if (this.step === 3) {
                if (!this.formData.loan_amount || this.formData.loan_amount <= 0) this.errors.loan_amount = 'Valid loan amount is required';
            }
            
            return Object.keys(this.errors).length === 0;
        },
        
        // Submit form
        submitForm() {
            if (!this.validateStep()) return;
            this.loading = true;
            document.getElementById('loanForm').submit();
        },
        
        // Format currency for display
        formatCurrency(value) {
            if (!value) return '₹0';
            return '₹' + parseInt(value).toLocaleString('en-IN');
        },
        
        // Get step title
        getStepTitle(step) {
            const titles = {
                1: 'Personal Information',
                2: 'Financial Information', 
                3: 'Loan Details',
                4: 'Review & Submit'
            };
            return titles[step] || '';
        },
        
        // Employment type label
        getEmploymentLabel(type) {
            const labels = {
                'salaried': 'Salaried',
                'self_employed': 'Self Employed',
                'unemployed': 'Unemployed',
                'student': 'Student',
                'retired': 'Retired'
            };
            return labels[type] || type;
        },
        
        // Home ownership label
        getOwnershipLabel(type) {
            const labels = {
                'own': 'Own',
                'mortgage': 'Mortgage',
                'rent': 'Rent',
                'other': 'Other'
            };
            return labels[type] || type;
        },
        
        // Loan purpose label
        getPurposeLabel(type) {
            const labels = {
                'home': 'Home Purchase',
                'car': 'Car Purchase',
                'education': 'Education',
                'business': 'Business',
                'personal': 'Personal',
                'medical': 'Medical',
                'debt_consolidation': 'Debt Consolidation'
            };
            return labels[type] || type;
        },
        
        // Loan tenure label
        getTenureLabel(months) {
            if (months >= 12) return Math.floor(months/12) + ' year' + (months >= 24 ? 's' : '') + ' (' + months + ' months)';
            return months + ' months';
        },
        
        // Computed debt-to-income ratio
        get debtToIncomeRatio() {
            if (!this.formData.annual_income || this.formData.annual_income <= 0) return 0;
            return ((this.formData.existing_debt || 0) / this.formData.annual_income * 100).toFixed(1);
        }
    };
}
